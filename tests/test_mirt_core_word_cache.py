"""No-fit reproduction of imported MIRT report references changing Word metadata.

Toy producer objects exercise recording only; no real calibration or new
psychometric evidence is produced.
"""
import pandas as pd
from core import reproducible as R
from core.academic_screen_results import adapt_screen,roots


def test_cached_mirt_report_cannot_add_null_producer_to_word_settings():
    page="Çok Boyutlu MTK (MIRT)"
    data=pd.DataFrame({"M1":[0,1,0,1],"M2":[1,0,1,0]})
    frame=pd.DataFrame({"Item":["M1","M2"],"a1":[1.,0.],"a2":[0.,1.],"d":[.1,-.1]})
    def producer(data,model):
        return {"coded":data.copy(),"model":model}
    def derived(outcome):
        return {"metadata":{"Converged":True},"tables":{"Parameters":frame.copy()}}
    original=R.Recorder(page)
    outcome=original.invoke(producer,(data,"2PL"),{},identity=("core.irt_multidim","fit"))
    result=original.invoke(derived,(outcome,),{},identity=("core.irt_multidim","report"))
    before=adapt_screen(page,original,[("Parameters",result["tables"]["Parameters"])],{},[],data)
    # Real reruns import recipes; only the displayed report product is remembered.
    # The old fit product is not magically reconstructed without fitting again.
    cached=R.Recorder(page,previous=original)
    index=cached._import_call(1,original)
    R._remember(result,[index],cached.references,cached.keep)
    assert 0 not in roots(cached) and 1 in roots(cached)
    after=adapt_screen(page,cached,[("Parameters",result["tables"]["Parameters"])],{},[],data)
    assert before==after
    assert before["metadata"]["settings"]==[{"API":"core.irt_multidim.fit","ayarlar":{"model":"2PL"}}]
    # Exact output changes still invalidate the academic envelope.
    changed=frame.copy();changed.loc[0,"d"]+=.01
    assert after!=adapt_screen(page,cached,[("Parameters",changed)],{},[],data)
