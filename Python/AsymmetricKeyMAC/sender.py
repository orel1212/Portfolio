from Crypto.PublicKey import RSA
from Crypto import Random
from Crypto.Signature.pkcs1_15 import PKCS115_SigScheme
from Crypto.Hash import SHA256
import socket
import sys
import json
import base64


def rsakeys():
    length = 2048
    privatekey = RSA.generate(length, Random.new().read)
    publickey = privatekey.publickey()
    f = open('keypair.pem', 'wb')
    f.write(private.export_key('PEM'))
    f.close()
    f = open('pubkey.pem', 'wb')
    f.write(public.export_key('PEM'))
    f.close()


def sign(keyPair, data):
    data = data.encode("utf8")
    signer = PKCS115_SigScheme(keyPair)
    hash = SHA256.new(data)
    return signer.sign(hash)


import os.path
from os import path

if path.exists("keypair.pem"):
    f2 = open('keypair.pem', 'r')
    privatekey = RSA.import_key(f2.read())
else:
    print("The private key file does not exist")
    print("Creating new keys...")
    rsakeys()
    print("Please share the generated public key with the other side and run again")
    exit()

# Bind the socket to the port
ip = '158.177.22.29'
port = 10002

try:
    if len(sys.argv) > 1:
        ip = str(sys.argv[1])
        if len(sys.argv) > 2:
            port = int(sys.argv[2])
except Exception as e:
    print(e)
    exit()

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = (ip, port)
print('connecting to {} port {}'.format(*server_address))
sock.connect(server_address)
print("Connected succesfully")

try:
    # Send data
    message = 'This is the message.'
    signature = sign(privatekey, message)
    data = json.dumps({"message": message, "signature": json.dumps(list(signature))})
    sock.send(data.encode())
    print('The message: {} and the signature were sent'.format(message))
except Exception as e:
    print(e)
