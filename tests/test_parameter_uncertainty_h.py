"""Independent transformed-objective derivatives and exact constraints; no fitting."""
import numpy as np
import pytest
from scipy.special import expit
from validation.parameter_uncertainty_h.runner import coordinates,transformed_covariance,G
from validation.binary_certification.reference_numpy import objective,BOX


@pytest.mark.parametrize("model",["3PL","4PL"])
def test_actual_transformed_objective_two_step_gradient_hessian(model):
    x=np.array([[0,1,0],[1,0,1],[1,1,1],[0,0,0]],float)
    f,qs,k=objective(x,model,points=61)
    p={"a":np.array([.8,1.2,1.5]),"b":np.array([-1.,0.,1.]),
       "c":np.array([.1,.15,.2]),"d":np.full(3,.93)}
    v=np.array([p[q][j] for j in range(k) for q in qs])
    limits=np.array([BOX[q] for j in range(k) for q in qs])
    eta,D,h=coordinates(v,limits)
    decode=lambda e:limits[:,0]+(limits[:,1]-limits[:,0])*expit(e)
    F=lambda e:f(decode(e))[0]  # NO density Jacobian: same physical MAP objective
    gp=f(v)[1]  # gradient of negative objective
    H=np.column_stack([(f(v+np.eye(len(v))[i]*1e-5)[1]-f(v-np.eye(len(v))[i]*1e-5)[1])/2e-5 for i in range(len(v))])
    chain=D[:,None]*H*D[None,:]+np.diag(gp*h)
    for step in (1e-4,2e-4):
        eye=np.eye(len(v))*step
        grad=np.array([(F(eta+e)-F(eta-e))/(2*step) for e in eye])
        np.testing.assert_allclose(grad,gp*D,atol=2e-6,rtol=2e-6)
        numeric=np.empty_like(chain)
        for i in range(len(v)):
            for j in range(len(v)):
                numeric[i,j]=(F(eta+eye[i]+eye[j])-F(eta+eye[i]-eye[j])-F(eta-eye[i]+eye[j])+F(eta-eye[i]-eye[j]))/(4*step*step)
        np.testing.assert_allclose(numeric,chain,atol=3e-5,rtol=3e-5)


def test_compressed_uncentered_meat_is_exact_and_prior_is_not_meat():
    patterns=np.array([[1.,2.],[-1.,0.],[2.,-1.]])
    frequency=np.array([3,2,4])
    expanded=np.repeat(patterns,frequency,axis=0)
    V=np.array([[.5,.1],[.1,.8]])
    D=np.array([.6,.8]);h=np.array([.1,-.1]);g=np.array([.2,-.1])
    C,A,B=transformed_covariance(V,patterns,frequency,D,h,g)
    np.testing.assert_allclose(B,(expanded*D).T@(expanded*D),atol=1e-14)
    np.testing.assert_allclose(A,D[:,None]*np.linalg.inv(V)*D[None,:]-np.diag(g*h),atol=1e-14)
    assert abs(B[0,1])>0 and np.linalg.eigvalsh(C)[0]>0


def test_native_box_transform_always_preserves_four_pl_order():
    for ec in (-100.,0.,100.):
        for ed in (-100.,0.,100.):
            c=BOX["c"][0]+(BOX["c"][1]-BOX["c"][0])*expit(ec)
            d=BOX["d"][0]+(BOX["d"][1]-BOX["d"][0])*expit(ed)
            assert 0<c<=.5<=d<1
    # All finite moderate transformed coordinates are strictly ordered.
    assert BOX["c"][0]+(BOX["c"][1]-BOX["c"][0])*expit(10)<.5
    assert .5<BOX["d"][0]+(BOX["d"][1]-BOX["d"][0])*expit(-10)
