import numpy as np
import matplotlib.pyplot as plt
import time
from math import gcd

def is_prime(n):
    """Check if a number is prime"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def generate_primes_upto_n_bits(max_bits):
    """Generate a map of lists of primes for each bit length up to max_bits"""
    primes_map = {}
    for bits in range(1, max_bits + 1):
        start = 1 << (bits - 1)
        end = (1 << bits) - 1
        primes = [n for n in range(start, end + 1) if is_prime(n)]
        if primes:
            primes_map[bits] = primes
    return primes_map

def generate_composites(primes_map, n_bits):
    """Generate composite numbers of n_bits by multiplying primes"""
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
    return list(composites)

def get_period(a, N):
    """Find period r such that a^r mod N = 1"""
    for r in range(1, N):
        if pow(a, r, N) == 1:
            return r
    return None

def shor_algorithm(N):
    """Perform Shor's algorithm on N and calculate success rate"""
    success = 0
    total = 0
    for a in range(2, N):
        total += 1
        if gcd(a, N) != 1:
            continue
        r = get_period(a, N)
        if r is None or r % 2 != 0:
            continue
        candidate = pow(a, r // 2, N)
        factor1 = gcd(candidate + 1, N)
        factor2 = gcd(candidate - 1, N)
        if 1 < factor1 < N or 1 < factor2 < N:
            success += 1
    success_rate = success / total if total > 0 else 0
    return success_rate

def measure_time(N):
    """Measure time taken to perform Shor's algorithm over all possible guesses of 'a'"""
    start_time = time.time()
    for a in range(2, N):
        if gcd(a, N) != 1:
            continue
        r = get_period(a, N)
        if r is None or r % 2 != 0:
            continue
        candidate = pow(a, r // 2, N)
        factor1 = gcd(candidate + 1, N)
        factor2 = gcd(candidate - 1, N)
        # Proceed to next 'a' without breaking, as we're timing the full algorithm
    end_time = time.time()
    return end_time - start_time

def test_shor_algorithm(max_bits=12):
    # Generate a map of primes for each bit length up to max_bits
    primes_map = generate_primes_upto_n_bits(max_bits)

    # Lists to store the results for plotting
    bit_sizes = []
    success_rates = []
    times = []

    # Iterate over each bit size from 2 to max_bits
    for n_bits in range(2, max_bits + 1):
        # Generate composite numbers with n_bits bits
        composites = generate_composites(primes_map, n_bits)
        if not composites:
            continue  # Skip if no composites are found

        # Lists to store success rates and times for current bit size
        composite_success_rates = []
        composite_times = []

        # Iterate over each composite number
        for N in composites:
            # Calculate the success rate of Shor's algorithm for N
            success_rate = shor_algorithm(N)
            # Measure the time taken for Shor's algorithm on N
            time_taken = measure_time(N)
            # Append the results to the lists
            composite_success_rates.append(success_rate)
            composite_times.append(time_taken)

        # Record the average success rate and time for current bit size
        bit_sizes.append(n_bits)
        success_rates.append(np.mean(composite_success_rates))
        times.append(np.mean(composite_times))

    # Create a figure with two subplots
    plt.figure(figsize=(12, 5))

    # Plot the success rates
    plt.subplot(1, 2, 1)
    plt.plot(bit_sizes, success_rates, marker='o')
    plt.xlabel('Number of Bits')
    plt.ylabel('Success Rate')
    plt.title("Shor's Algorithm Success Rate")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(bit_sizes, times, marker='o')
    plt.xlabel('Number of Bits')
    plt.ylabel('Average Time (s)')
    plt.title("Shor's Algorithm Time Taken")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_shor_algorithm(max_bits=8)