"""No-fit installed Python mirt API/source audit."""
import inspect,json,pathlib
import mirt
root=pathlib.Path(mirt.__file__).parent
print("PYTHON_MIRT",mirt.__version__,flush=True)
for name in dir(mirt):
    if any(t in name.lower() for t in ("m2","fit_stat","gof","lrt","likelihood_ratio","compare")):
        obj=getattr(mirt,name)
        print("PUBLIC_API",name,str(inspect.signature(obj)) if callable(obj) else type(obj).__name__,flush=True)
for file in sorted(root.rglob("*.py")):
    src=file.read_text()
    if any(t in src for t in ("def M2","def m2","def compute_m2","def limited","RMSEA","SRMSR","def likelihood_ratio")):
        print("RUNTIME_SOURCE_BEGIN",file.relative_to(root),flush=True)
        print(src[:65000],flush=True)
        print("RUNTIME_SOURCE_END",file.relative_to(root),flush=True)
