import pandas as pd
import numpy as np
from core.irt_se_coverage import interval_coverage_study

def test_coverage_reports_denominators_even_when_hessian_unavailable():
    # 4PL needs k>=5 (20 <= 2**5-1); with k=3 every run would be a structural error.
    truth=pd.DataFrame({"Madde":["M1","M2","M3","M4","M5"],
                        "a":[.8,1.2,1.5,1.,1.3],"b":[-.6,0.,.7,-.2,.4],
                        "c":[.1,.12,.14,.11,.13],"d":[.91,.93,.95,.92,.94]})
    result=interval_coverage_study(
        truth,"4PL",sample_sizes=(100,),repetitions=1,
        seed=71,points=11,max_iter=2,starts=(0.,))
    assert len(result["runs"])==1
    assert result["summary"]["Tekrar"].iloc[0]==1
    assert 0<=result["summary"]["SH hesaplanabilirlik oranı"].iloc[0]<=1
    assert set(result["runs"]["Durum"]).issubset({
        "Yakınsamadı","SH reddedildi","SH hesaplandı"})
    assert "Kapsama(a)" in result["summary"].columns
    assert result["summary"]["İncelenen aralık(a)"].iloc[0] in (0,5)


def test_structurally_unidentified_design_is_reported_not_estimated():
    truth=pd.DataFrame({"Madde":["M1","M2","M3"],
                        "a":[.8,1.2,1.5],"b":[-.6,0.,.7],
                        "c":[.1,.12,.14],"d":[.91,.93,.95]})
    result=interval_coverage_study(truth,"4PL",sample_sizes=(100,),repetitions=1,
                                   seed=71,points=11,max_iter=2,starts=(0.,))
    assert result["runs"]["Durum"].tolist()==["Hata"]
    assert "yapısal olarak tanımlanamaz" in result["runs"]["Neden"].iloc[0]
    assert result["summary"]["SH hesaplanabilen"].iloc[0]==0
