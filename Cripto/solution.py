import random

def pseudo_random_generator(seed, length):
    a = 1664525
    c = 1013904223
    m = 2 ** 32
    x = seed
    buf = []
    while len(buf) < length:
        x = (a * x + c) % m
        b = bin(x)[2:]
        for ch in b:
            buf.append(1 if ch == '1' else 0)
            if len(buf) == length:
                break
    return buf[:length]

def Gen(G, seed, n):
    return G(seed, n)


def _bits_to_text_msb(bits):
    out = []
    i = 0
    while i + 8 <= len(bits):
        byte = bits[i:i+8]
        out.append(chr(int(byte, 2)))  # byte es '01000001' -> 'A'
        i += 8
    return "".join(out)

def Enc(k, m):
    # k: bitstring de longitud 8*len(m)
    # m: texto (string)
    m_bits = ""
    for ch in m:
        m_bits += format(ord(ch), "08b")

    L = min(len(k), len(m_bits))
    ct_bits = []
    for i in range(L):
        ct_bits.append(str(int(m_bits[i]) ^ int(k[i])))

    return _bits_to_text_msb("".join(ct_bits))


def _text_to_bitstring_msb(s):
    # convierte texto a bits MSB-first por byte
    return "".join(format(ord(ch), "08b") for ch in s)

def _normalize_k_to_bitstring(k):
    # acepta k como bitstring o como lista de 0/1
    if isinstance(k, str):
        return k
    return "".join("1" if int(b) else "0" for b in k)

def Dec(k, c):
    # k: bitstring (o lista 0/1) con longitud >= 8*len(c)
    # c: string (texto) devuelto por Enc
    k_bits = _normalize_k_to_bitstring(k)
    c_bits = _text_to_bitstring_msb(c)

    L = min(len(k_bits), len(c_bits))
    m_bits = []
    i = 0
    while i < L:
        m_bits.append("1" if (int(c_bits[i]) ^ int(k_bits[i])) else "0")
        i += 1

    return _bits_to_text_msb("".join(m_bits))






def D1(y):
    # G(x) = x || OR(x1,...,xn)
    # El ultimo bit de la salida siempre es 1, salvo que la entrada sea 0^n.
    # Eso significa que la distribucion no puede ser uniforme, ya que
    # en un string aleatorio de n+1 bits el ultimo bit seria 0 o 1 con probabilidad 1/2.
    # El distinguidor revisa simplemente si el ultimo bit es 1 siempre.
    if isinstance(y, str):
        bits = [1 if ch == '1' else 0 for ch in y]
    else:
        bits = [1 if int(b) != 0 else 0 for b in y]

    if len(bits) == 0:
        return False
    last = bits[-1]
    or_prev = 1 if any(bits[:-1]) else 0
    return (last == or_prev)


def D2(y):
    # G(x) = x || XOR(x1,...,xn)
    # El ultimo bit es la paridad de los n primeros bits.
    # En un string uniforme, la paridad seria 0 o 1 con probabilidad 1/2,
    # pero en la salida de G siempre se cumple la relacion "ultimo bit = XOR de los anteriores".
    # El distinguidor puede comprobar esa condicion y detectar que no es uniforme.
    bits = [int(b) for b in y[:-1]]
    last = int(y[-1])
    return (last == (sum(bits) % 2))

def D3(y):
    # G(x) = x || x
    # La salida consiste en repetir dos veces el mismo bloque x.
    # En una cadena uniforme de 2n bits, la probabilidad de que
    # las dos mitades sean exactamente iguales es 2^-n (negligible).
    # En la salida de G ocurre siempre. El distinguidor compara
    # las dos mitades y si son iguales acepta.
    mid = len(y) // 2
    return (y[:mid] == y[mid:])



def A_messages(n):
    m0 = [0 for _ in range(n)]
    m1 = [1 for _ in range(n)]
    return m0, m1

def A_attack(m0, m1, c):
    # estrategia: revisar el ultimo bit del ciphertext
    # como el generador sesga a que k[-1] = 1,
    # si c[-1] == 1 significa que m0 fue encriptado (adversario devuelve 0).
    if c[-1] == 1:
        return 0
    else:
        return 1


def A_messages_multiple(n):
    m0 = [[0 for _ in range(n)] for _ in range(2)]
    m1 = [[1 for _ in range(n)] for _ in range(2)]
    return m0, m1

def A_attack_multiple(m0, m1, c):
    # c: lista de ciphertexts (cada uno lista de ints 0/1)
    ones = 0
    for ci in c:
        if ci[-1] == 1:
            ones += 1
    zeros = len(c) - ones
    # Si predominan unos al final, inferimos m0 (bit 0); si predominan ceros, m1 (bit 1).
    # En caso de empate, favorecemos 0 por el sesgo del OR (keystream con ultimo bit 1).
    return 0 if ones >= zeros else 1
