Q1) The Probability of Shooting at a Target
Model 1: Uniform over area
Total area of disk = 100π
Area within 1 unit of boundary:
π(10
2
−9
2
)=19π
P(A)=
100π
19π
	​

=0.19
Model 2: Radius uniformly distributed
R∼Uniform(0,10)
P(A)=P(9≤R≤10)=
10
1
	​

=0.1
Explanation

The results differ because:

Uniform over area gives higher weight to outer regions
Uniform over radius treats all radii equally

Thus, the probability distributions are different.

Q2) Bertrand’s Paradox

Probability that a random chord is longer than the side of an inscribed equilateral triangle:

Different models give:
Random endpoints on circle:
P=
3
1
	​

Random midpoint in circle:
P=
4
1
	​

Random radius method:
P=
2
1
	​

Conclusion

Different definitions of “random chord” lead to different probabilities.

Q8) Probability of Being a Girl

Sample space:

{BB,BG,GB,GG}

Given at least one girl:

{BG,GB,GG}
P(both girls)=
3
1
	​

Q9) Information in a Discrete Random Variable
Probability space
Ω={G1,G2,D1,D2},F=P(Ω)
σ-algebras
σ(X)={∅,Ω,{G1,G2},{D1,D2}}
σ(Y)=P(Ω)
σ(X)⊆σ(Y)
Marginal distributions

X:

P(X=0)=0.7,P(X=1)=0.3

Y:

(0.5,0.2,0.1,0.2)
Joint probabilities
P(X=0,Y=1)=0.5
P(X=0,Y=2)=0.2
P(X=1,Y=3)=0.1
P(X=1,Y=4)=0.2
Conditional probabilities
P(X=1∣Y=3)=1,P(X=1∣Y=4)=1
P(Y=3∣X=1)=
3
1
	​

,P(Y=4∣X=1)=
3
2
	​

Interpretation
Conditioning on Y gives full information
Conditioning on X gives partial information
Entropy
H(X)≈0.61
H(Y)≈1.22
Python Visualization
import matplotlib.pyplot as plt

labels = ['G1','G2','D1','D2']
probs = [0.5,0.2,0.1,0.2]

plt.bar(labels, probs)
plt.title("Distribution of Y")
plt.show()
Q10) Continuous Random Variables
Probability space
Ω=R
2
,F=B(R
2
)
Joint density
f(t,p)=
4π
1
	​

e
−t
2
/2−p
2
/8
σ-algebras
σ(X)⊆σ(Y)
Marginals
X∼N(0,1)
P∼N(0,4)
Probabilities
P(X≤0)=0.5
P(X>1)≈0.1587
P(Y∈(−∞,0]×R)=0.5
P(Y∈[−1,1]×[−2,2])≈0.46
Conditional distributions
X∣Y=(t,p)=t
Y∣X=t=(t,P), where P∼N(0,4)
Measurability
X=π
1
	​

∘Y
Entropy
h(X)=
2
1
	​

ln(2πe)
h(Y)=
2
1
	​

ln(2πe)+
2
1
	​

ln(8πe)
h(Y∣X)=
2
1
	​

ln(8πe)
h(X∣Y)=−∞
Interpretation
Conditioning on X: only temperature known
Conditioning on Y: full system known
Continuous case allows infinite precision → negative infinite conditional entropy
