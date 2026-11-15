import pytest

from tools.check_disco_notebook_parity import schedule_literal,saved_literals,compare_range_wiring


def test_reads_notebook_crop_schedules_without_evaluating_code():
    values=schedule_literal('[16]*400+[12]*200+[8]*200+[4]*200')
    assert len(values)==1000
    assert [values[index] for index in (0,399,400,599,600,799,800,999)]==[16,16,12,12,8,8,4,4]


@pytest.mark.parametrize('expression',[
    "__import__('os').system('echo bad')",'[1]*100000000','[1]*-1','[1]+1','1','[x for x in range(10)]',
])
def test_rejects_executable_and_unbounded_schedule_expressions(expression):
    with pytest.raises(ValueError): schedule_literal(expression)


def test_settings_extraction_records_literals_not_runtime_overrides():
    saved=saved_literals([(0,"init_scale=0\nframes_scale=15000\ndef run():\n init_scale=args.frames_scale\n")])
    assert saved['init_scale']=={'value':0,'cell':0,'line':1}
    assert saved['frames_scale']['value']==15000


def test_range_wiring_audit_distinguishes_disabled_and_effective_loss():
    disabled,enabled=compare_range_wiring()
    assert disabled['gradient']['exact']
    assert enabled['notebook_rms']==0 and enabled['application_rms']>0
