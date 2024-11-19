from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np
from math import gcd, pi
from fractions import Fraction
from Crypto.Util.number import bytes_to_long, long_to_bytes, getPrime, inverse, GCD


def inverse_qft(qc, qubits):
    """Custom implementation of inverse QFT"""
    n = len(qubits)

    # Swap qubits
    for i in range(n // 2):
        qc.swap(qubits[i], qubits[n - i - 1])

    # Apply inverse QFT operations
    for j in range(n):
        for k in range(j):
            qc.cp(-pi / float(2 ** (j - k)), qubits[k], qubits[j])
        qc.h(qubits[j])


def get_period(a: int, N: int, n_count: int, shots=100):
    """Find the period of a^r mod N using quantum circuit"""

    n_count = max(n_count, N.bit_length())
    counting_qubits = QuantumRegister(n_count, "count")
    phase_qubits = QuantumRegister(n_count, "phase")
    c = ClassicalRegister(n_count)
    qc = QuantumCircuit(counting_qubits, phase_qubits, c)

    # Initialize counting qubits in superposition
    for qubit in counting_qubits:
        qc.h(qubit)

    # Apply controlled-U operations
    for i in range(len(counting_qubits)):
        power = pow(a, 2**i, N)
        controlled_mod_mult(qc, counting_qubits[i], phase_qubits, power, N)

    # Apply inverse QFT using our custom implementation
    inverse_qft(qc, counting_qubits)

    # Measure counting qubits
    qc.measure(counting_qubits, c)

    # Execute circuit
    backend = AerSimulator()
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Process results to find period
    measured_phases = []
    for output in counts:
        decimal = int(output, 2)
        phase = decimal / (2**n_count)
        measured_phases.append(phase)

    # Find period from measured phases
    for phase in measured_phases:
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator
        if r % 2 == 0 and pow(a, r, N) == 1:
            return r
    return None


def controlled_mod_mult(qc, control, target, a, N):
    """Applies controlled modular multiplication by a mod N"""
    n = len(target)
    for i in range(n):
        qc.cp(2 * np.pi * a * (2**i) / N, control, target[i])


def shor_algorithm(N: int) -> list:
    """Main Shor's algorithm implementation"""
    # print(f"Starting factorization of {N}")
    if N % 2 == 0:
        return [2, N // 2]

    # Check if N is prime power
    for b in range(2, int(np.sqrt(N)) + 1):
        if pow(b, 2, N) == 1:
            return [b, N // b]

    # Try random values of a
    for attempt in range(50):  # Limit attempts
        # print(f"\nAttempt {attempt + 1}:")
        a = np.random.randint(2, N)
        # print(f"Chosen random a: {a}")
        if gcd(a, N) != 1:
            factor = gcd(a, N)
            # print(f"Found factor through GCD: {factor}")
            return [factor, N // factor]

        # print(f"Finding period for a={a}")
        r = get_period(a, N, N.bit_length() + 2)
        # print(f"Found period r: {r}")
        if r is None or r % 2 != 0:
            continue

        candidate = pow(a, r // 2, N)
        factor1 = gcd(candidate + 1, N)
        factor2 = gcd(candidate - 1, N)
        # print(f"GCD results: {factor1}, {factor2}")

        if factor1 != 1 and factor1 != N:
            return [factor1, N // factor1]
        if factor2 != 1 and factor2 != N:
            return [factor2, N // factor2]

    return None


# # Test the implementation
# if __name__ == "__main__":
#     N = 589  # Number to factor
#     factors = shor_algorithm(N)
#     if factors:
#         print(f"Factors of {N} are: {factors}")
#     else:
#         print(f"Failed to factor {N}")
def is_prime(n):
    """Check if number is prime"""
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def test_shor_algorithm(max_N=255):
    """Test Shor's algorithm on composite numbers up to user input, default is 255"""
    print(f"Test Shor's algorithm on composite numbers up to {max_N}")
    # Generate prime numbers up to sqrt(255)
    primes = [n for n in range(2, int(np.sqrt(max_N)) + 1) if is_prime(n)]

    # Generate composite numbers that are products of two primes
    composites = []
    for i in range(len(primes)):
        for j in range(i, len(primes)):
            product = primes[i] * primes[j]
            if product <= max_N:
                composites.append(product)

    composites = sorted(list(set(composites)))  # Remove duplicates and sort

    # Test each composite number
    results = []
    for N in composites:
        print(f"Testing N = {N}")
        # print("=" * 40)

        start_time = time.time()
        factors = shor_algorithm(N)
        end_time = time.time()

        success = factors is not None and factors[0] * factors[1] == N

        result = {
            "N": N,
            "factors": factors,
            "success": success,
            "time": end_time - start_time,
        }
        results.append(result)

        # print(f"Result: {'Success' if success else 'Failed'}")
        # print(f"Factors: {factors}")
        # print(f"Time taken: {result['time']:.2f} seconds")

    # Print summary
    print("\nSummary")
    print("=" * 40)
    successes = sum(1 for r in results if r["success"])
    print(f"Total tests: {len(results)}")
    print(f"Successful factorizations: {successes}")
    print(f"Success rate: {(successes/len(results))*100:.1f}%")

    # Print detailed results
    print("\nDetailed Results:")
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"{status} N={r['N']}: factors={r['factors']} ({r['time']:.2f}s)")


def extended_gcd(a, b):
    """Extended Euclidean Algorithm"""
    if a == 0:
        return b, 0, 1
    else:
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y


def modinv(e, phi):
    """Compute the modular inverse of e modulo phi"""
    gcd, x, _ = extended_gcd(e, phi)
    if gcd != 1:
        raise Exception("Modular inverse does not exist")
    else:
        return x % phi


# Custom small RSA key generation
class CustomRSA:
    def __init__(self, bit_length=8):
        self.p = getPrime(bit_length // 2)
        self.q = getPrime(bit_length // 2)
        self.n = self.p * self.q
        self.phi = (self.p - 1) * (self.q - 1)

        # Choose e such that 1 < e < φ(n) and GCD(e, φ(n)) == 1
        self.e = 3
        while GCD(self.e, self.phi) != 1:
            self.e += 2  # Try the next odd number

        self.d = inverse(self.e, self.phi)

    def init2(self, p, q, e, bit_length=8):
        self.p = p
        self.q = q
        self.n = self.p * self.q
        self.phi = (self.p - 1) * (self.q - 1)
        self.e = e
        self.d = inverse(self.e, self.phi)

    def encrypt(self, plaintext):
        m = bytes_to_long(plaintext)
        if m >= self.n:
            raise ValueError("Plaintext too large for the key size")
        c = pow(m, self.e, self.n)
        return long_to_bytes(c)

    def decrypt(self, ciphertext):
        c = bytes_to_long(ciphertext)
        m = pow(c, self.d, self.n)
        return long_to_bytes(m)


class SimpleSymmetricEncryption:
    def __init__(self, key):
        self.key = key  # Key should be a single byte for simplicity

    def encrypt(self, plaintext):
        """Encrypts the plaintext using a simple XOR operation with the key."""
        return bytes([b ^ self.key for b in plaintext])

    def decrypt(self, ciphertext):
        """Decrypts the ciphertext using a simple XOR operation with the key."""
        return bytes([b ^ self.key for b in ciphertext])


if __name__ == "__main__":
    import time

    # Input the hex value to get the e and N
    hex_message = input("Enter a hexadecimal value for RSA encryption: ")
    hex_message = int(hex_message, 16)

    # Input to get the symmetric key
    symmetric_key_encrypted = input("Enter the encrypted symmetric key e.g. 0x7f : ")
    symmetric_key_encrypted = int(symmetric_key_encrypted, 16)

    # Input the encrypted message to decrypt
    encrypted_message = input("Enter the encrypted message you want to decrypt: ")
    encrypted_message = int(encrypted_message, 16)

    # Convert the input string to a hexadecimal integer
    byte_array = hex_message.to_bytes(3, byteorder='big')

    # Extract the individual bytes
    first_byte = byte_array[0]
    middle_byte = byte_array[1]
    last_byte = byte_array[2]

    # Convert the first and last bytes to integers
    first_integer = first_byte
    last_integer = last_byte

    pipe_char = middle_byte

    N = first_integer
    e = last_integer

    factors = shor_algorithm(N)

    rsa = CustomRSA()

    rsa.init2(factors[0], factors[1], e)
    symmetric_key = bytes_to_long(rsa.decrypt(long_to_bytes(symmetric_key_encrypted)))
    symmetric_encryption = SimpleSymmetricEncryption(symmetric_key)
    ciphertext = encrypted_message.to_bytes((encrypted_message.bit_length() + 7) // 8, "big")

    decrypted_bytes = symmetric_encryption.decrypt(ciphertext)
    decrypted_message = decrypted_bytes.decode('utf-8', errors='ignore')

    print("\nDecryption Results:")
    print("===================")
    print(f"Symmetric key: {symmetric_key}")
    print(f"Decrypted message: {decrypted_message}")
    print(f"Factors of {N} are: {factors}")

    # SimpleSymmetricEncryption symmetric_encrpytion = SimpleSymmetricEncryption(symmetric_key)
    # encrypted_plaintext = b"Hello, World!"
    # message = symmetric_encrpytion.decrypt(plaintext)

    # print(f"Decrypted message: {message}")
