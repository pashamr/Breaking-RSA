import os
import random
import socket
import threading
from Crypto.Util.number import bytes_to_long, long_to_bytes, getPrime, inverse, GCD


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

    def init2(self, p,q,e, bit_length=8):
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


# Simple symmetric key encryption and decryption
class SimpleSymmetricEncryption:
    def __init__(self, key):
        self.key = key  # Key should be a single byte for simplicity

    def encrypt(self, plaintext):
        """Encrypts the plaintext using a simple XOR operation with the key."""
        return bytes([b ^ self.key for b in plaintext])

    def decrypt(self, ciphertext):
        """Decrypts the ciphertext using a simple XOR operation with the key."""
        return bytes([b ^ self.key for b in ciphertext])


# Socket-based communication
choice = input("Do you want to host (1) or to connect (2): ")

if choice == "1":
    rsa = CustomRSA(bit_length=8)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 1500))
    server.listen()
    client, _ = server.accept()

    # Step 1: Send public key (n, e) to the client
    client.send(long_to_bytes(rsa.n) + b"|" + long_to_bytes(rsa.e))

    # Step 2: Receive encrypted symmetric key from client
    encrypted_symmetric_key = client.recv(1024)
    symmetric_key = int.from_bytes(
        rsa.decrypt(encrypted_symmetric_key), byteorder="big"
    )
    print("Received symmetric key:", symmetric_key)

    sym_enc = SimpleSymmetricEncryption(symmetric_key)

elif choice == "2":
    rsa = CustomRSA(bit_length=8)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 1500))

    # Step 1: Receive public key (n, e) from server
    n, e = client.recv(1024).split(b"|")
    n = bytes_to_long(n)
    e = bytes_to_long(e)
    print("Received public key (n, e):", n, e)

    server_public_key = CustomRSA(bit_length=8)
    server_public_key.n = n
    server_public_key.e = e

    # Step 2: Generate symmetric key and send encrypted version to server
    symmetric_key = random.randint(1, server_public_key.n)
    print("Generated symmetric key:", symmetric_key)
    encrypted_symmetric_key = server_public_key.encrypt(long_to_bytes(symmetric_key))
    print(e)
    client.send(encrypted_symmetric_key)

    sym_enc = SimpleSymmetricEncryption(symmetric_key)

else:
    exit()


# Functions for sending and receiving messages
def sending_messages(sock, symmetric_enc):
    while True:
        message = input("")
        if message.lower() == "exit":
            sock.send(symmetric_enc.encrypt(b"exit"))
            print("You have left the chat.")
            sock.close()
            break
        encrypted_message = symmetric_enc.encrypt(message.encode())
        sock.send(encrypted_message)
        print("You (encrypted):", encrypted_message.hex())


def receiving_messages(sock, symmetric_enc):
    while True:
        encrypted_message = sock.recv(1024)
        decrypted_message = symmetric_enc.decrypt(encrypted_message)
        if decrypted_message.decode().lower() == "exit":
            print("Partner has left the chat.")
            sock.close()
            break
        print("Partner:", decrypted_message.decode())


# Start threads for communication
threading.Thread(target=sending_messages, args=(client, sym_enc)).start()
threading.Thread(target=receiving_messages, args=(client, sym_enc)).start()
