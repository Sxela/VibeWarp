"""Shared model download queue with independent pause/cancel/resume controls."""
import copy
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from vibewarp.acquisition import DownloadEvent, acquire, download_scope, install_download_hooks
from vibewarp.cancellation import cancellation_scope, raise_if_cancelled
from vibewarp.model_assets import selected_assets, models_root

ACTIVE = {'queued','starting','downloading','waiting','verifying','retrying','pausing','cancelling'}


@dataclass
class ModelDownload:
    name: str
    path: str
    operation: object = None
    kind: str = 'asset'
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = 'queued'
    downloaded: int = 0
    total: int | None = None
    speed: float = 0.
    error: str = ''
    active_file: str = ''
    request: str | None = None
    owners: set = field(default_factory=set)
    runtime_job: str | None = None

    def snapshot(self):
        return {key:getattr(self,key) for key in
                ('id','name','path','status','downloaded','total','speed','error','active_file')} | {
            'can_pause':self.status in ACTIVE and self.status not in ('pausing','cancelling'),
            'can_cancel':self.status in ACTIVE or self.status == 'paused',
            'can_resume':self.status == 'paused' or (self.kind != 'runtime' and self.status in ('cancelled','failed'))}


@dataclass
class Preparation:
    manager: object
    item_ids: list
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def snapshot(self):
        return self.manager.batch_snapshot(self)


class DownloadManager:
    def __init__(self, workers=2):
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._tasks, self._keys, self._items, self._sources = {}, {}, {}, {}
        self._workers, self._started = max(1,workers), False

    def _specs(self, config):
        specs=[]
        for asset in selected_assets(config):
            specs.append(((asset.path,asset.url,asset.sha256),Path(asset.path).name,asset.path,'asset',
                lambda asset=asset:acquire(asset.url,asset.path,sha256=asset.sha256)))
        if config.model_version.startswith('disco_') and (config.disco.sampler != 'distilled' or config.disco.distilled_guided):
            from vibewarp.disco_catalog import MODERN_ENCODERS
            from vibewarp.core.disco import ensure_guidance_assets
            for name in config.disco.encoders:
                if name in MODERN_ENCODERS:
                    single=copy.deepcopy(config)
                    single.disco.encoders=[name]
                    path=str((models_root(config)/'disco'/name).resolve())
                    specs.append(((path,MODERN_ENCODERS[name]),name,path,'bundle',
                                  lambda single=single:ensure_guidance_assets(single)))
        return specs

    def prepare(self, config, owner=None):
        specs=self._specs(copy.deepcopy(config))
        key=tuple(spec[0] for spec in specs)
        with self._wake:
            previous=self._keys.get(key)
            if previous and previous.snapshot()['state'] in ('queued','running','paused'):
                task=previous
            else:
                ids=[]
                for signature,name,path,kind,operation in specs:
                    item=self._sources.get(signature)
                    if item is None:
                        item=ModelDownload(name,path,operation,kind)
                        self._items[item.id]=item
                        self._sources[signature]=item
                    elif item.status in ('ready','cancelled','failed'):
                        item.status,item.request,item.error='queued',None,''
                    ids.append(item.id)
                task=Preparation(self,ids)
                self._tasks[task.id]=task
                self._keys[key]=task
            consumer=owner or f'manual:{task.id}'
            for item_id in task.item_ids:
                self._items[item_id].owners.add(consumer)
            if not self._started:
                self._started=True
                for index in range(self._workers):
                    threading.Thread(target=self._worker,name=f'vibewarp-download-{index}',daemon=True).start()
            self._wake.notify_all()
            return task

    def get(self, task_id):
        with self._lock:
            return self._tasks.get(task_id)

    def batch_snapshot(self, task):
        with self._lock:
            files=[self._items[key].snapshot() for key in task.item_ids]
            states={file['status'] for file in files}
            if not states or states == {'ready'}: state='completed'
            elif states & ACTIVE: state='running'
            elif 'paused' in states: state='paused'
            elif 'failed' in states: state='failed'
            else: state='cancelled'
            return {'id':task.id,'state':state,'files':files,
                    'error':'; '.join(file['error'] for file in files if file['error'])}

    def snapshot(self):
        with self._lock:
            priority={'downloading':0,'verifying':0,'starting':0,'waiting':0,'retrying':0,
                      'pausing':0,'cancelling':0,'queued':1,'paused':2,'failed':3,'cancelled':4,'ready':5}
            items=sorted(self._items.values(),key=lambda item:priority.get(item.status,3))
            return {'files':[item.snapshot() for item in items],
                    'tasks':[task.snapshot() for task in self._tasks.values()]}

    def _control(self, item, action):
        if action == 'pause' and item.status in ACTIVE:
            item.request='paused'
            item.status='paused' if item.status == 'queued' or item.kind == 'runtime' else 'pausing'
        elif action == 'cancel' and item.status in ACTIVE|{'paused'}:
            item.request='cancelled'
            item.status='cancelled' if item.status in ('queued','paused') or item.kind == 'runtime' else 'cancelling'
        elif action == 'resume' and item.snapshot()['can_resume']:
            item.request,item.error=None,''
            item.owners.add(f'manual:{item.id}')
            item.status='downloading' if item.kind == 'runtime' else 'queued'
        else:
            raise ValueError(f'Cannot {action} a {item.status} download')
        item.speed=0.

    def control(self, item_id, action):
        if action not in ('pause','resume','cancel'):
            raise ValueError('Unknown download action')
        with self._wake:
            item=self._items.get(item_id)
            if item is None: raise KeyError(item_id)
            self._control(item,action)
            self._wake.notify_all()
            return item.snapshot()

    def cancel_batch(self,task_id):
        with self._wake:
            task=self._tasks[task_id]
            for key in task.item_ids:
                item=self._items[key]
                if item.status in ACTIVE|{'paused'}:
                    self._control(item,'cancel')
            self._wake.notify_all()
            return task.snapshot()

    def _worker(self):
        while True:
            with self._wake:
                self._wake.wait_for(lambda:any(item.operation is not None and item.status == 'queued' for item in self._items.values()))
                item=next(item for item in self._items.values() if item.operation is not None and item.status == 'queued')
                item.status='starting'
            def progress(event):
                with self._lock:
                    item.downloaded,item.total,item.speed=event.downloaded,event.total,event.speed
                    item.active_file=event.name if item.kind == 'bundle' else ''
                    if item.request:
                        item.status='pausing' if item.request == 'paused' else 'cancelling'
                    else:
                        item.status=('downloading' if item.kind == 'bundle' else 'verifying') if event.status == 'ready' else event.status
            outcome, error = 'ready', ''
            try:
                with cancellation_scope(lambda:item.request is not None),download_scope(progress):
                    install_download_hooks()
                    item.operation()
            except KeyboardInterrupt:
                outcome='cancelled'
            except Exception as exc:
                outcome,error='failed',str(exc)
            finally:
                with self._wake:
                    # Publish the terminal state only after releasing the transfer's
                    # resources; a retry may immediately claim this queue entry.
                    item.status=item.request or outcome
                    item.error='' if item.request else error
                    item.speed=0.
                    if item.status != 'paused': item.owners.clear()
                    self._wake.notify_all()

    def wait_for_render(self,config,owner,progress):
        task=self.prepare(config,owner)
        seen={}
        try:
            while True:
                raise_if_cancelled()
                snapshot=task.snapshot()
                for row in snapshot['files']:
                    signature=(row['status'],row['downloaded'],row['total'],row['error'])
                    if seen.get(row['id']) != signature:
                        seen[row['id']]=signature
                        progress(DownloadEvent(row['name'],row['path'],row['downloaded'],row['total'],row['status'],row['speed'],row['error']))
                if snapshot['state'] not in ('queued','running','paused'):
                    if snapshot['state'] != 'completed':
                        stopped=', '.join(f"{row['name']}: {row['status']}" for row in snapshot['files'] if row['status'] != 'ready')
                        raise RuntimeError(f'Required model downloads did not complete: {stopped}')
                    return
                with self._wake: self._wake.wait(timeout=.1)
        finally:
            with self._wake:
                for key in task.item_ids:
                    item=self._items[key]
                    item.owners.discard(owner)
                    if not item.owners and item.status in ACTIVE|{'paused'}:
                        self._control(item,'cancel')
                self._wake.notify_all()

    def runtime_progress(self,event,job_id):
        """Controls for lazy transfers discovered while a model is loading."""
        with self._wake:
            if any(item.kind != 'runtime' and (item.path == event.path or
                   (item.kind == 'bundle' and Path(event.path).resolve().is_relative_to(Path(item.path).resolve())))
                   for item in self._items.values()):
                return
            key=('runtime',job_id,event.path)
            item=self._sources.get(key)
            if item is None:
                item=ModelDownload(event.name,event.path,kind='runtime',runtime_job=job_id)
                self._items[item.id]=item
                self._sources[key]=item
            item.downloaded,item.total,item.speed=event.downloaded,event.total,event.speed
            if item.request != 'paused' and item.request != 'cancelled':
                item.status=event.status
            while item.request == 'paused':
                self._wake.wait(timeout=.1)
                raise_if_cancelled()
            if item.request == 'cancelled':
                raise RuntimeError(f'Download cancelled: {item.name}')

    def finish_runtime(self,job_id,state):
        with self._wake:
            for item in self._items.values():
                if item.runtime_job == job_id and item.status in ACTIVE|{'paused'}:
                    item.status='cancelled' if state == 'cancelled' else 'failed'
            self._wake.notify_all()
