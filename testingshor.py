from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np
from math import gcd, pi
from fractions import Fraction
import time
import matplotlib.pyplot as plt

def inverse_qft(qc, qubits):
    """Custom implementation of inverse QFT"""
    n = len(qubits)
    
    # Swap qubits
    for i in range(n//2):
        qc.swap(qubits[i], qubits[n-i-1])
    
    # Apply inverse QFT operations
    for j in range(n):
        for k in range(j):
            qc.cp(-pi/float(2**(j-k)), qubits[k], qubits[j])
        qc.h(qubits[j])

def get_period(a: int, N: int, n_count: int, shots=2000):
    """Improved period finding function"""
    n_count = max(n_count, N.bit_length() + 3)  # Extra qubits for precision
    
    counting_qubits = QuantumRegister(n_count, 'count')
    phase_qubits = QuantumRegister(1, 'phase')  # Simplified to 1 phase qubit
    c = ClassicalRegister(n_count)
    qc = QuantumCircuit(counting_qubits, phase_qubits, c)

    # Initialize phase qubit to |1⟩
    qc.x(phase_qubits[0])

    # Initialize counting register to superposition
    for qubit in counting_qubits:
        qc.h(qubit)

    # Apply controlled operations
    for i in range(n_count):
        power = pow(a, 2**i, N)
        controlled_mod_mult(qc, counting_qubits[i], phase_qubits, power, N)

    # Apply inverse QFT
    inverse_qft(qc, counting_qubits)
    qc.measure(counting_qubits, c)

    # Run circuit
    backend = AerSimulator()
    job = backend.run(qc, shots=shots)
    counts = job.result().get_counts()

    # Process results
    max_count = 0
    best_bitstring = None
    
    for bitstring, count in counts.items():
        if count > max_count:
            max_count = count
            best_bitstring = bitstring

    if best_bitstring is None:
        return None

    # Convert measurement to phase
    phase = int(best_bitstring, 2) / (2**n_count)
    
    # Find period using continued fractions
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator

    # Verify the period
    if r > 0 and pow(a, r, N) == 1:
        return r

    return None

def controlled_mod_mult(qc, control, target, a, N):
    """Simplified controlled modular multiplication"""
    n = len(target)
    # Use simpler phase rotation
    phase = 2 * np.pi * a / N
    qc.cp(phase, control, target[0])

def find_coprimes(N):
    """Find all numbers that are coprime with N."""
    coprimes = []
    for a in range(2, N):
        if gcd(a, N) == 1:
            coprimes.append(a)
    return coprimes

def shor_algorithm(N: int) -> list:
    """Main Shor's algorithm implementation"""
    # print(f"Starting factorization of {N}")
    if N % 2 == 0:
        return [2, N//2]
    
    # Check if N is prime power
    for b in range(2, int(np.sqrt(N)) + 1):
        if pow(b, 2, N) == 1:
            return [b, N//b]

    # Get list of coprimes
    coprimes = find_coprimes(N)
    if not coprimes:
        return None

    # Try each coprime value of a
    for a in coprimes:  # Limit to first 30 coprimes for efficiency
        # print(f'a is {a}')
        r = get_period(a, N, N.bit_length() + 2)
        # print(f'period is {r}')
        if r is None or r % 2 != 0:
            continue
            
        candidate = pow(a, r//2, N)
        factor1 = gcd(candidate + 1, N)
        factor2 = gcd(candidate - 1, N)
        
        if factor1 != 1 and factor1 != N:
            return [factor1, N//factor1]
        if factor2 != 1 and factor2 != N:
            return [factor2, N//factor2]
    
    return None


def is_prime(n):
    """Check if number is prime"""
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def generate_primes_upto_n_bits(max_bits):
    """Generate a map of lists of primes for each bit length up to max_bits."""
    primes_map = {}
    for bits in range(2, max_bits + 1):
        start = 1 << (bits - 1)
        end = (1 << bits) - 1
        primes = [n for n in range(start, end + 1) if is_prime(n)]
        if primes:
            primes_map[bits] = primes
    return primes_map

def generate_composites(primes_map, n_bits):
    """Generate composite numbers of n_bits by multiplying primes."""
    composites = set()
    target_start = 1 << (n_bits - 1)
    target_end = (1 << n_bits) - 1
    for bits1 in primes_map:
        for bits2 in primes_map:
            for p1 in primes_map[bits1]:
                for p2 in primes_map[bits2]:
                    composite = p1 * p2
                    if target_start <= composite <= target_end:
                        composites.add(composite)
    return sorted(list(composites))

def test_quantum_shor(max_bits=8):
    """Test quantum Shor's algorithm implementation."""
    print(f"Testing Quantum Shor's Algorithm up to {max_bits} bits")
    
    # Generate primes and composites
    primes_map = generate_primes_upto_n_bits(max_bits)
    
    # Lists to store results
    bit_sizes = []
    success_rates = []
    times = []
    
    for n_bits in range(2, max_bits + 1):
        print(f"\nTesting {n_bits}-bit numbers...")
        
        # Generate composites for current bit size
        composites = generate_composites(primes_map, n_bits)
        if not composites:
            print(f"No composite numbers found for {n_bits} bits")
            continue
            
        successful_factorizations = 0
        total_time = 0
        
        for N in composites:
            #print(f"  Testing N = {N}")
            
            # Measure time and success of factorization
            start_time = time.time()
            factors = shor_algorithm(N)
            end_time = time.time()
            
            # Record results
            time_taken = end_time - start_time
            total_time += time_taken
            
            if factors and factors[0] * factors[1] == N:
                successful_factorizations += 1
                #print(f"    Success: Found factors {factors}")
            #else:
                #print(f"    Failed to find factors")
                
        # Calculate statistics for current bit size
        success_rate = successful_factorizations / len(composites)
        avg_time = total_time / len(composites)
        
        bit_sizes.append(n_bits)
        success_rates.append(success_rate * 100)  # Convert to percentage
        times.append(avg_time)
        
        print(f"Bit size {n_bits} summary:")
        print(f"  Success rate: {success_rate*100:.2f}%")
        print(f"  Average time: {avg_time:.3f} seconds")
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Plot success rates
    plt.subplot(1, 2, 1)
    plt.plot(bit_sizes, success_rates, marker='o')
    plt.xlabel('Number of Bits')
    plt.ylabel('Success Rate (%)')
    plt.title("Quantum Shor's Algorithm Success Rate")
    plt.grid(True)
    
    # Plot timing
    plt.subplot(1, 2, 2)
    plt.plot(bit_sizes, times, marker='o')
    plt.xlabel('Number of Bits')
    plt.ylabel('Average Time (s)')
    plt.title("Quantum Shor's Algorithm Time Taken")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run full test suite
    test_quantum_shor(max_bits=10)
