import pandas as pd
from core.irt_simulation_reporting import annotate_generating_model, export_recovery_excel
from io import BytesIO

def test_recovery_labels_separate_truth_from_misspecified_fit():
    study={
        "summary":pd.DataFrame({"Model":["3PL","4PL"],"N":[200,200]}),
        "runs":pd.DataFrame({"Model":["3PL","4PL"],"N":[200,200]}),
        "item_recovery":pd.DataFrame({"Model":["3PL","4PL"],"Madde":["M1","M1"]}),
        "metadata":{}
    }
    result=annotate_generating_model(study,"3PL")
    for key in ("summary","runs","item_recovery"):
        frame=result[key]
        assert frame["Veri üreten model"].tolist()==["3PL","3PL"]
        assert frame["Model koşulu"].tolist()==[
            "Doğru model","Yanlış tanımlanmış model"]
    assert result["metadata"]["generating_model"]=="3PL"
    wb=pd.read_excel(BytesIO(export_recovery_excel(result)),sheet_name="Özet")
    assert wb["Model koşulu"].tolist()==[
        "Doğru model","Yanlış tanımlanmış model"]
