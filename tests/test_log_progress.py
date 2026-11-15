import pytest

from vibewarp.log_progress import render_eta
from vibewarp.web_jobs import Job, JobManager, _LineWriter
from vibewarp.config import RunConfig


@pytest.mark.parametrize('line,expected',[
    ('Rendering frames: 1%| | 1/101 [00:56<1:34:06, 56.46s/it, time=56.5',5646),
    ('\x1b[2K\rRendering frames: 75%|###| 3/4 [03:00<00:56, 56.46s/it]',56),
    ('Contact sheets: 2/3 [00:56<02:03, 60s/it]',123),
    ('AnimateDiff batches: 1/3 [01:00<25:00:01, 0.5it/s]',90001),
    ('Rendering frames: 100%|#| 101/101 [00:56<00:00, 56.46s/it]',0),
    ('Rendering frames: 0%| | 0/101 [00:56<?, ?it/s, time=56.5s]',None),
    ('Rendering frames: 1/4 [00:56<00:99, 1it/s]',None),
])
def test_outer_tqdm_eta(line,expected):
    assert render_eta(line)==(True,expected)


@pytest.mark.parametrize('line',[
    'Disco sampling frame 0: steps=6, strength=0.666667, executed=4, seed=2551302960',
    'frame 1 steps: 1/4 [00:10<00:30, 10s/step]',
    'WARNING: mask not found', '[11:12:31] diffusion render (frame 0): 53.6s',
    'Downloading weights: 1/10 [00:01<00:09, 1it/s]',
])
def test_inner_progress_and_unrelated_logs_do_not_replace_render_eta(line):
    assert render_eta(line)==(False,None)


def test_job_eta_survives_unrelated_lines_and_is_hidden_on_terminal_or_cancel():
    manager=JobManager()
    job=Job(id='example',config=RunConfig(),state='running',stage='rendering')
    manager._log(job,'Rendering frames: 1/101 [00:56<1:34:06, 56.46s/it]')
    manager._log(job,'frame 1 steps: 1/4 [00:10<00:30, 10s/step]')
    manager._log(job,'WARNING: a different warning')
    assert job.snapshot()['eta_seconds']==5646
    assert job.snapshot()['eta_updated_at']
    job.cancel_requested=True
    assert job.snapshot()['eta_seconds'] is None
    job.cancel_requested=False
    job.stage='cleanup'
    assert job.snapshot()['eta_seconds'] is None
    job.stage=job.state='completed'
    assert job.snapshot()['eta_seconds'] is None


def test_carriage_return_updates_and_split_writes_are_individual_log_lines():
    lines=[]
    writer=_LineWriter(lines.append)
    text='\rRendering frames: 1/101 [00:56<1:34:06, 56s/it]'
    assert writer.write(text)==len(text)
    writer.write('\rRendering frames: 2/101 [01:52<1:33:')
    writer.write('10, 56s/it]\nwarning\r\n')
    writer.flush()
    assert lines==[text.lstrip('\r'),'Rendering frames: 2/101 [01:52<1:33:10, 56s/it]','warning']


def test_writer_routes_redraws_to_progress_callback_and_crlf_is_one_line():
    lines,progress=[],[]
    writer=_LineWriter(lines.append,progress.append)
    writer.write('loading model\r')
    writer.write('\n')  # CRLF split across writes is one ordinary line
    writer.write('\r 10%|#  | 5/50\r 20%|## | 10/50\r100%|###| 50/50\n')
    writer.write('\x1b[2K\rRendering frames: 1/4 [00:01<00:03]\n')
    writer.write('done\r\n')
    writer.flush()
    assert lines==['loading model','done']
    assert progress==[' 10%|#  | 5/50',' 20%|## | 10/50','100%|###| 50/50',
                      'Rendering frames: 1/4 [00:01<00:03]']


def test_progress_redraws_replace_their_own_line_and_keep_real_logs():
    manager=JobManager()
    job=Job(id='example',config=RunConfig(),state='running',stage='rendering')
    writer=_LineWriter(lambda line: manager._log(job,line),
                       lambda line: manager._log(job,line,progress=True))
    writer.write('Run directory: out/run-1\n')
    for frame in range(1,30):
        writer.write(f'\rRendering frames: {frame}/101 [00:56<1:34:06, 56s/it]')
        for step in range(1,51):
            writer.write(f'\r{step*2}%|#| {step}/50 [00:01<00:01]')
        writer.write('\n')
        writer.write(f'saved frame {frame}\n')
    writer.flush()
    assert job.logs[0]=='Run directory: out/run-1'
    assert job.logs.count('saved frame 29')==1
    # One line per bar, updated in place, plus the real lines.
    assert len(job.logs)==1+2+29
    assert job.logs[1]=='Rendering frames: 29/101 [00:56<1:34:06, 56s/it]'
    assert job.snapshot()['eta_seconds']==5646


def test_progress_line_trimmed_out_of_window_is_appended_again():
    manager=JobManager()
    job=Job(id='example',config=RunConfig())
    manager._log(job,'bar: 1/9',progress=True)
    for index in range(600):
        manager._log(job,f'line {index}')
    manager._log(job,'bar: 2/9',progress=True)
    assert len(job.logs)==500 and job.logs[-1]=='bar: 2/9'
    manager._log(job,'bar: 3/9',progress=True)
    assert job.logs[-1]=='bar: 3/9' and 'bar: 2/9' not in job.logs
