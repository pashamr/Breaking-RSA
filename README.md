# Breaking-RSA
Simulates a quantum computer using qiskit and implements Shor's Algorithm to try and break RSA Encryption. Due to the limitations of simulation, the bit size of RSA that can be broken is restricted.

## Shor’s Algorithm 

Shor’s Algorithm focuses on solving the problem, “Given an odd composite number N, find its integer factors”. Its steps involve both classical and quantum computing, as the strengths of quantum computing are only used in figuring out the period of modular exponentiation. To begin with, it assumes that N is not a prime power and N’s factors are non-trivial, as those numbers can be efficiently solved using classical algorithms.

### General Steps:

1. Let $N = pq$, where $p$ and $q$ are prime numbers
    
2. Let there be $a$, where $1 < a < N$ and $gcd(a, N) = 1$
    
3. Let r b the period of modular exponentiation of $a^xmod(N)$
    
4. With a good approximation of r, the  $gcd(a^\frac{r}{2}- 1, N)$ and $gcd(a^\frac{r}{2}+ 1, N)$ has a good chance of containing $p$ and/or $q$
  

### Classical Steps in Shor’s Algorithm

In selecting a random a, which is a coprime of N, we prepare for the modular arithmetic portion of the algorithm. If $gcd(a, N) >  1$, this means that the factor has already been found. 

  

### Quantum Computing in Shor’s Algorithm

In Step 3 of the algorithm, we have to figure out the period of modular exponentiation following axmod(N). This period is the smallest possible r such that, 

$$ar ≡ 1 mod(N)$$

Which can be expanded to,

$$ar - 1 ≡ 0 mod(N)$$

$$ar - 1 = N$$

$$N =(a^\frac{r}{2}- 1)(a^\frac{r}{2} + 1)$$
As seen above, if we can figure out $r$, then we can solve for the factors of $N$.
  

To solve this period efficiently, we utilise quantum computing’s property to store the repeating results of $a^xmod(N)$ in a single superposition with phase factors proportional to x, as defined below.


$$
|u_1\rangle = \frac{1}{\sqrt{r}} \sum_{k=0}^{r-1} e^{-\frac{2\pi i k}{r}} |a^k \mod N\rangle
$$

  

When we expand this superposition and sum up these eigenstates, the different phases will cancel each other out, due to the repeating pattern r, thus leaving only the basis state of ∣1​⟩.  

$$
\frac{1}{\sqrt{r}} \sum_{s=0}^{r-1} |u_s\rangle = |1\rangle
$$

Once we have this eigenstate, we can perform Quantum Phase Estimation. It does this by encoding the phase r to the amplitudes of a quantum register, and then extracting the phase through an inverse Quantum Fourier Transform.

The quantum circuit used to represent the algorithm is as follows.


![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdqtc09UrK83RDEws0tXYkLd0T50_Ccm2Gyt37jfi6nPC8nxLsgIrTCP8xTcnICEqNbVwkUOHEP01ePwSTUrjIjTGcqFYXJbpwMo18MxnbmyGRbgjQCIc4CzjfxvlDTMEB_U9qt?key=VKkuDdmB8Hfo3whuTA9K53Q3)


At first, it applies a Hadamard transform to the phase register to create a uniform superposition. Afterwards, it encodes r to the quantum register by applying the controlled powers of the Unitary operator,  U. Lastly, the inverse Quantum Fourier Transform translates the phase-encoded amplitudes into sharp peaks corresponding to the phase ϕ. By measuring the register, we get the binary approximation of ϕ.

 Through the Quantum Phase Estimation, we get the following phase:

$$\Phi = \frac{s}{r}$$

Where $( 0 < s < r - 1 )$.  

Afterwards, we can use the continued fractions algorithm to find r as an integer, thus figuring out the period of modular exponentiation, and solving the two factors of N.

In our source code, we implemented each of these steps using Qiskit’s quantum circuit & quantum computing simulator (Aer)



Language: Python

## Dependencies

This project requires the following Python libraries:

- qiskit
- qiskit-aer
- numpy
- math
- fractions
- pycryptodome

```bash
pip install qiskit qiskit-aer numpy pycryptodome```

## Usage
1. Run the script on two different terminals 

```bash
python main.py```

Make sure that one terminal acts as host and another connects as client

2. Input message in client's terminal 

3. Sniff packets from wireshark and get the public key, symmetric key, and the encrypted message

4. Run the shor's algorithm script

```bash
python shor.py```

5. Input the public key, symmetric key, and encrypted message from wireshark

6. Shor's algorithm script should output factors of N, decrypted symmetric key, and final decrypted message

