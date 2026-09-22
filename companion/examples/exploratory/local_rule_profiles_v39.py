"""New exploratory rule-24/44 diagnostic, not the frozen E3 experiment or a repair.

All contexts are equally weighted. These are finite local truth tables, not the
eight full-ring E3 probes. No files, fits, held-out tests or priority claims.
"""
from fractions import Fraction
from itertools import product
import json

def bit_rule(rule,left,centre,right):
    return (rule>>(4*left+2*centre+right))&1

def boolean_rule(rule,left,centre,right):
    if rule==24:
        return int((left and not centre and not right) or (not left and centre and right))
    if rule==44:
        return int((not left and centre) or (left and not centre and right))
    raise ValueError(rule)

def shrink(rule,context,horizon):
    cells=tuple(context)
    for _ in range(horizon):
        cells=tuple(bit_rule(rule,*cells[i:i+3]) for i in range(len(cells)-2))
    assert len(cells)==1
    return cells[0]

def boolean_array(rule,context,horizon):
    # Independent synchronous finite-array route, padded beyond the causal cone.
    cells=[0]*4+list(context)+[0]*4
    for _ in range(horizon):
        following=cells.copy()
        for i in range(1,len(cells)-1):
            following[i]=boolean_rule(rule,cells[i-1],cells[i],cells[i+1])
        cells=following
    return cells[len(cells)//2]

def influence(rule,horizon):
    n=2*horizon+1;contexts=list(product((0,1),repeat=n))
    counts=[0]*n
    for context in contexts:
        original=shrink(rule,context,horizon)
        assert original==boolean_array(rule,context,horizon)
        for i in range(n):
            changed=list(context);changed[i]=1-changed[i]
            other=shrink(rule,changed,horizon)
            assert other==boolean_array(rule,changed,horizon)
            counts[i]+=original!=other
    return tuple(Fraction(c,len(contexts)) for c in counts)

def recoded(rule,reflect,complement):
    code=0
    for L,C,R in product((0,1),repeat=3):
        a,b,c=(R,C,L) if reflect else (L,C,R)
        value=1-bit_rule(rule,1-a,1-b,1-c) if complement else bit_rule(rule,a,b,c)
        code|=value<<(4*L+2*C+R)
    return code

def bit_influence(rule):
    contexts=list(product((0,1),repeat=3));result=[]
    for i in range(3):
        total=0
        for c in contexts:
            x=list(c);x[i]=1-x[i]
            total+=bit_rule(rule,*c)!=bit_rule(rule,*x)
        result.append(Fraction(total,8))
    return result

def main():
    expected={24:((Fraction(1,2),)*3,(Fraction(3,8),Fraction(3,8),Fraction(3,8),Fraction(1,8),Fraction(1,8))),
              44:((Fraction(3,4),Fraction(3,4),Fraction(1,4)),(Fraction(7,16),Fraction(5,16),Fraction(7,16),Fraction(3,16),Fraction(1,16)))}
    results={}
    for rule in (24,44):
        one,two=influence(rule,1),influence(rule,2)
        assert (one,two)==expected[rule]
        mean_one=(one[0]+one[2])/2;mean_two=sum(two[i] for i in (0,1,3,4))/4
        assert mean_one==Fraction(1,2) and mean_two==Fraction(1,4)
        imbalance=abs(one[0]-one[2]);transforms=[]
        for reflection,complement in product((False,True),repeat=2):
            transformed=recoded(rule,reflection,complement);profile=bit_influence(transformed)
            assert abs(profile[0]-profile[2])==imbalance
            transforms.append(dict(reflected=reflection,complemented=complement,rule=transformed,profile=list(map(str,profile))))
        results[str(rule)]=dict(one_step_left_centre_right=list(map(str,one)),one_step_neighbour_mean=str(mean_one),
            reflection_invariant_imbalance=str(imbalance),two_step_minus2_to_plus2=list(map(str,two)),
            two_step_noncentral_mean=str(mean_two),recodings=transforms)
    return dict(passed=True,status='new exploratory companion exercise, not adopted as a book result',
        origin='Prompted by the v38 external reviewer after examining the exposed pair. Independently implemented here.',
        convention='f_r(L,C,R) = (r >> (4L+2C+R)) & 1',
        context_measure='Uniform over all 8 one-step and all 32 two-step local contexts',
        horizons=[1,2],implementations='Bit-table shrinking window and explicit Boolean rules on a padded synchronous array',
        results=results,original_e3_rerun=False,terminal_response_predicted=False,held_out_cases_tested=False,
        limitations=['Not Conway Game of Life.','Not E3 full-ring preparations or response variable.',
            'Distinguishes this exposed pair only; no causal explanation of their E3 difference.',
            'No fitted replacement, transfer success, general descriptor validation or novelty claim.'])

if __name__=='__main__':
    print(json.dumps(main(),indent=2))
