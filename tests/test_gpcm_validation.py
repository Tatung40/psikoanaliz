"""No-fit GPCM identities, parameter/gradient mapping and posterior safeguards."""
import numpy as np
import pandas as pd
import pytest
import mirt
from scipy.special import logsumexp
from validation import grm_r_comparison as G
from core.irt_person_abilities import person_abilities


def test_gpcm_likelihood_and_both_scores():
    a=np.array([.8,1.7]);b=np.array([[.5,-.5,.2],[-1.,.3,1.2]])
    theta=np.array([-2.,-1.,0.,1.,2.]);w=np.array([.05,.2,.5,.2,.05])
    x=np.array([[0,1],[2,3],[1,2],[3,0]])
    ll,g=G.marginal(x,a,b,theta,w,'GPCM')
    model=mirt.GeneralizedPartialCredit(2,4);model.set_parameters(discrimination=a,steps=b)
    P=model.probability(theta)
    direct=sum(np.log(sum(w[q]*np.prod([P[q,i,row[i]] for i in range(2)]) for q in range(5))) for row in x)
    np.testing.assert_allclose(ll,direct,rtol=1e-13)
    for name,arr in [('a',a),('b',b)]:
        for ix in np.ndindex(arr.shape):
            ap,am,bp,bm=a.copy(),a.copy(),b.copy(),b.copy()
            up,dn=(ap,am) if name=='a' else (bp,bm);up[ix]+=1e-5;dn[ix]-=1e-5
            diff=(G.marginal(x,ap,bp,theta,w,'GPCM')[0]-G.marginal(x,am,bm,theta,w,'GPCM')[0])/2e-5
            np.testing.assert_allclose(g[name][ix],diff,rtol=1e-6,atol=1e-8)
    pcm=G.marginal(x,np.ones(2),b,theta,w,'PCM')[0]
    gpcm=G.marginal(x,np.ones(2),b,theta,w,'GPCM')[0]
    assert pcm==gpcm  # Nested fixed slope, NOT all GPCM slopes fixed.


def scoring_fixture():
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    from core.irt_b14_model_curves import fitted_category_curves
    from types import SimpleNamespace
    q=GaussHermiteQuadrature(n_points=61,n_dimensions=1)
    m=mirt.GeneralizedPartialCredit(3,4)
    m.set_parameters(discrimination=np.array([.8,1.2,1.7]),steps=np.array([[.5,-.5,.2],[-1,0,1],[0,1,2]]))
    curves=fitted_category_curves(dict(result=SimpleNamespace(model=m),k=3,items=['A','B','C'],labels=['0','1','2','3']),q.nodes[:,0])
    x=pd.DataFrame([[0,1,2],[3,2,1],[1,0,3]],columns=['A','B','C'],index=[7,7,2])
    return x,curves,(q.nodes[:,0],q.weights)


def test_person_grid_identity_item_order_and_ids():
    x,c,q=scoring_fixture();ids=['p3','p1','p2']
    got=person_abilities(x,c,model='GPCM',identifiers=ids,quadrature=q)
    shuffled=person_abilities(x[['C','A','B']],c.sample(frac=1,random_state=4),model='GPCM',identifiers=ids,quadrature=q)
    pd.testing.assert_frame_equal(got,shuffled)
    assert got.Birey.tolist()==ids and got['Ham puan'].tolist()==x.sum(1).tolist()
    lp=np.log(q[1]).copy()
    for j,item in enumerate(x.columns):
        P=c[(c.Madde==item)&(c.Kategori==str(x.iloc[0,j]))].sort_values('Theta').Olasılık.to_numpy()
        lp+=np.log(P)
    w=np.exp(lp-logsumexp(lp));mean=w@q[0];sd=np.sqrt(w@((q[0]-mean)**2))
    np.testing.assert_allclose(got.iloc[0][['Yetenek (θ)','SH (θ)']].to_numpy(float),[mean,sd],atol=1e-13)
    with pytest.raises(ValueError,match='benzersiz'):person_abilities(x,c,model='GPCM',identifiers=['p','p','q'],quadrature=q)
    with pytest.raises(ValueError,match='madde sırası'):person_abilities(x,c,model='GPCM',item_names=['B','A','C'],quadrature=q)
    with pytest.raises(ValueError,match='Quadrature'):person_abilities(x,c,model='GPCM',quadrature=(q[0],-q[1]))


def test_impossible_patterns_and_nominal_order_fail_closed():
    x,c,q=scoring_fixture();c=c.copy()
    c.loc[c.Kategori=='0','Olasılık']=1.;c.loc[c.Kategori!='0','Olasılık']=0.
    with pytest.raises(ValueError,match='olasılığı sıfır'):person_abilities(x,c,model='GPCM',quadrature=q)
    c=scoring_fixture()[1].replace({'Kategori':{'0':'Z','1':'B','2':'A','3':'K'}})
    with pytest.raises(ValueError,match='kod sırası'):person_abilities(x,c,model='NRM',quadrature=q)
    got=person_abilities(x,c,model='NRM',category_labels=['Z','B','A','K'],quadrature=q)
    assert got['Ham puan'].isna().all()
