import threading
import time
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from vibewarp.acquisition import DownloadEvent, report
from vibewarp.cancellation import cancellation_scope, raise_if_cancelled
from vibewarp.config import RunConfig
from vibewarp.model_assets import ModelAsset
from vibewarp.web_downloads import DownloadManager


def wait_until(check):
    deadline=time.monotonic()+3
    while not check():
        assert time.monotonic()<deadline, 'Queue did not reach expected state'
        time.sleep(.005)


@pytest.fixture
def queue(tmp_path,monkeypatch):
    manager=DownloadManager(workers=1)
    starts=[]
    release=threading.Event()
    assets=[ModelAsset(f'https://example.org/{name}',str(tmp_path/name)) for name in ('a.pt','b.pt','c.pt')]
    monkeypatch.setattr('vibewarp.web_downloads.selected_assets',lambda config:assets[:config.video.width])
    def acquire(url,path,**kwargs):
        starts.append(path)
        report(DownloadEvent(path.split('\\')[-1],path,5,10))
        if path==assets[0].path:
            while not release.wait(.005):
                raise_if_cancelled()
        report(DownloadEvent('weights',path,10,10,'ready'))
    monkeypatch.setattr('vibewarp.web_downloads.acquire',acquire)
    yield manager,assets,starts,release
    release.set()


def test_pause_frees_worker_for_next_model_and_resume_keeps_identity(queue):
    manager,assets,starts,release=queue
    task=manager.prepare(RunConfig())
    first,second,third=task.item_ids
    wait_until(lambda:manager.snapshot()['files'][0]['downloaded']==5)
    manager.control(third,'cancel')
    manager.control(first,'pause')
    wait_until(lambda:manager._items[first].status=='paused' and manager._items[second].status=='ready')
    assert manager._items[third].status=='cancelled'
    assert assets[2].path not in starts
    assert manager._items[first].downloaded==5
    release.set()
    manager.control(first,'resume')
    wait_until(lambda:manager._items[first].status=='ready')
    assert starts.count(assets[0].path)==2
    assert manager._items[second].status=='ready'


def test_queue_lists_multiple_batches_and_deduplicates_shared_models(queue):
    manager,assets,starts,release=queue
    config=RunConfig()
    config.video.width=1
    first=manager.prepare(config)
    config.video.width=2
    second=manager.prepare(config)
    assert first.item_ids[0]==second.item_ids[0]
    assert len(manager.snapshot()['files'])==2
    assert len(manager.snapshot()['tasks'])==2
    release.set()
    wait_until(lambda:second.snapshot()['state']=='completed')
    assert starts.count(assets[0].path)==1


def test_render_cancellation_keeps_manually_requested_download(queue):
    manager,assets,starts,release=queue
    config=RunConfig()
    config.video.width=1
    task=manager.prepare(config)
    stopped=threading.Event()
    errors=[]
    def render():
        try:
            with cancellation_scope(stopped.is_set):
                manager.wait_for_render(config,'render:test',lambda event:None)
        except KeyboardInterrupt:
            errors.append('cancelled')
    thread=threading.Thread(target=render)
    thread.start()
    wait_until(lambda:'render:test' in manager._items[task.item_ids[0]].owners)
    stopped.set()
    thread.join(2)
    assert errors==['cancelled']
    assert manager._items[task.item_ids[0]].request is None
    release.set()
    wait_until(lambda:task.snapshot()['state']=='completed')


def test_render_waits_for_paused_download_and_reports_progress(queue):
    manager,assets,starts,release=queue
    config=RunConfig()
    config.video.width=1
    task=manager.prepare(config)
    item=task.item_ids[0]
    wait_until(lambda:manager._items[item].downloaded==5)
    manager.control(item,'pause')
    wait_until(lambda:manager._items[item].status=='paused')
    done=threading.Event()
    events=[]
    thread=threading.Thread(target=lambda:(manager.wait_for_render(config,'render:test',events.append),done.set()))
    thread.start()
    wait_until(lambda:any(event.status=='paused' for event in events))
    assert not done.is_set()
    release.set()
    manager.control(item,'resume')
    thread.join(2)
    assert done.is_set()
    assert events[-1].status=='ready'


def test_global_queue_api_controls_one_model(queue,monkeypatch):
    from vibewarp.web import create_app
    manager,assets,starts,release=queue
    monkeypatch.setattr('vibewarp.web.download_manager',manager)
    client=TestClient(create_app())
    assert client.get('/api/models/downloads').json()['files']==[]
    response=client.post('/api/models/prepare',json=asdict(RunConfig()))
    assert response.status_code==200
    ids=[row['id'] for row in response.json()['files']]
    assert client.post(f'/api/models/downloads/{ids[2]}/pause').json()['status']=='paused'
    assert client.post(f'/api/models/downloads/{ids[2]}/cancel').json()['status']=='cancelled'
    assert manager._items[ids[0]].request is None
    assert client.post('/api/models/downloads/missing/cancel').status_code==404
    assert client.post(f'/api/models/downloads/{ids[0]}/invalid').status_code==422
    assert len(client.get('/api/models/downloads').json()['files'])==3


def test_lazy_download_pause_resume_and_cancel():
    manager=DownloadManager()
    event=DownloadEvent('lazy.pt','lazy.pt',5,10)
    manager.runtime_progress(event,'job')
    item=manager.snapshot()['files'][0]['id']
    manager.control(item,'pause')
    done=threading.Event()
    thread=threading.Thread(target=lambda:(manager.runtime_progress(event,'job'),done.set()))
    thread.start()
    assert not done.wait(.05)
    manager.control(item,'resume')
    thread.join(2)
    assert done.is_set()
    manager.control(item,'cancel')
    with pytest.raises(RuntimeError,match='Download cancelled'):
        manager.runtime_progress(event,'job')


def test_render_prepares_global_queue_before_entering_pipeline(queue,monkeypatch):
    from vibewarp.web_jobs import JobManager
    manager,assets,starts,release=queue
    entered=[]
    def run(config,**kwargs):
        entered.append([row['status'] for row in manager.snapshot()['files']])
        return []
    monkeypatch.setattr('vibewarp.pipeline.run',run)
    jobs=JobManager()
    jobs.download_manager=manager
    job=jobs.submit(RunConfig())
    wait_until(lambda:len(manager.snapshot()['files'])==3)
    assert not entered
    release.set()
    wait_until(lambda:job.state in ('completed','failed'))
    assert job.state=='completed',job.error
    assert entered==[['ready','ready','ready']]


def test_cancelling_render_cancels_its_exclusive_queue_entries(queue):
    from vibewarp.web_jobs import JobManager
    manager,assets,starts,release=queue
    jobs=JobManager()
    jobs.download_manager=manager
    job=jobs.submit(RunConfig())
    wait_until(lambda:len(manager.snapshot()['files'])==3)
    jobs.cancel(job.id)
    wait_until(lambda:job.state=='cancelled')
    wait_until(lambda:all(row['status']=='cancelled' for row in manager.snapshot()['files']))
