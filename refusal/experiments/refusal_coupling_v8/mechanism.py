"""Dense forward weights with either unrestricted or exactly sliced optimizer support."""
import math
import torch


class Support:
    def __init__(self,model,layout=None):
        self.model=model;self.layout=layout
        named=dict(model.named_parameters())
        self.names=list(named) if layout is None else [r['name']+'.weight' for r in layout]
        self.actual=[named[n] for n in self.names]
        self.axes=[None]*len(self.names) if layout is None else [r['axis'] for r in layout]
        self.indices=[None]*len(self.names) if layout is None else [
            torch.tensor(r['indices'][0],device=p.device) for r,p in zip(layout,self.actual)]
        self.enable()
        self.optparams=self.actual if layout is None else [
            torch.nn.Parameter(self.take(p,i).detach().clone()) for i,p in enumerate(self.actual)]
        self.count=sum(p.numel() for p in self.optparams)

    def enable(self):
        self.model.requires_grad_(False)
        for p in self.actual:p.requires_grad_(True)

    def take(self,value,i):
        return value if self.axes[i] is None else value.index_select(self.axes[i],self.indices[i])

    def values(self):
        return [self.take(p,i).detach().cpu().clone() for i,p in enumerate(self.actual)]

    def gradients(self):
        return [self.take(p.grad,i).detach().cpu().clone() if p.grad is not None else
                torch.zeros_like(self.take(p,i),device='cpu') for i,p in enumerate(self.actual)]

    def put(self,values):
        assert len(values)==len(self.actual)
        with torch.no_grad():
            for i,(p,v) in enumerate(zip(self.actual,values)):
                v=v.to(p.device)
                if self.axes[i] is None:p.copy_(v)
                else:p.index_copy_(self.axes[i],self.indices[i],v)

    def apply(self,opt,grads,clip=1.):
        self.model.zero_grad(set_to_none=True)
        if self.layout is not None:
            with torch.no_grad():
                for i,(p,v) in enumerate(zip(self.actual,self.optparams)):v.copy_(self.take(p,i))
        for p,g in zip(self.optparams,grads):
            assert torch.isfinite(g).all()
            p.grad=g.to(p.device)
        norm=torch.nn.utils.clip_grad_norm_(self.optparams,clip)
        assert torch.isfinite(norm)
        opt.step();opt.zero_grad(set_to_none=True)
        if self.layout is not None:self.put([p.detach() for p in self.optparams])
        return float(norm)


def dot(a,b):
    return sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b))


def gradient(support,loss):
    support.model.zero_grad(set_to_none=True)
    value=loss()
    assert torch.isfinite(value)
    value.backward()
    result=support.gradients()
    support.model.zero_grad(set_to_none=True)
    return result,float(value.detach())


def cosine_gradient(support,attack_loss,skill_loss,epsilon):
    """Forward finite differences of the two Hessian-vector products in grad cosine."""
    a,la=gradient(support,attack_loss);b,lb=gradient(support,skill_loss)
    na=max(math.sqrt(dot(a,a)),1e-12);nb=max(math.sqrt(dot(b,b)),1e-12)
    c=dot(a,b)/(na*nb);assert math.isfinite(c) and abs(c)<=1+1e-6
    va=[y/nb-c*x/na for x,y in zip(a,b)]
    vb=[x/na-c*y/nb for x,y in zip(a,b)]
    original=support.values()
    try:
        support.put([x+epsilon*v for x,v in zip(original,va)])
        shifted,_=gradient(support,attack_loss)
        result=[(x-y)/(epsilon*na) for x,y in zip(shifted,a)]
        support.put([x+epsilon*v for x,v in zip(original,vb)])
        shifted,_=gradient(support,skill_loss)
        for out,x,y in zip(result,shifted,b):out.add_((x-y)/(epsilon*nb))
    finally:
        support.put(original)
    return result,b,dict(attack_nll=la,skill_nll=lb,cosine=c,attack_grad_norm=na,skill_grad_norm=nb)
