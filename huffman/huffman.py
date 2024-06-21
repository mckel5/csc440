import marshal
import os
import pickle
import sys
from array import array
from typing import Dict
from typing import Tuple
from heapq import heapify, heappop, heappush


def encode(message: bytes) -> Tuple[str, Dict]:
    """Given the bytes read from a file, encodes the contents using the Huffman encoding algorithm.

    :param message: raw sequence of bytes from a file
    :returns: string of 1s and 0s representing the encoded message
              dict containing the decoder ring as explained in lecture and handout.
    """
    _, _, root_children = build_tree(message)

    # Determine codewords
    if not isinstance(root_children, tuple):
        # Handle messages composed of only one character
        codewords = {root_children: "0"}
    else:
        codewords = calculate_codewords(*root_children)

    encoded = ""

    # Build encoded string using codewords
    for byte in message:
        encoded += codewords[byte]

    # Pad the last byte with 0s if not all bits are used
    # Record the length of the padding (0 <= x < 8) using the first byte
    padding_length = 8 - (len(encoded) % 8)
    encoded = f"{padding_length:08b}" + encoded + ("0" * padding_length)

    # Decoder ring is simply the inverted codewords dict
    decoder_ring = {code: word for (word, code) in codewords.items()}

    return encoded, decoder_ring


def decode(message: str, decoder_ring: Dict) -> bytes:
    """Given the encoded string and the decoder ring, decodes the message using the Huffman decoding algorithm.

    :param message: string of 1s and 0s representing the encoded message
    :param decoder_ring: dict containing the decoder ring
    return: raw sequence of bytes that represent a decoded file
    """
    padding_length = int(message[0:8], 2)

    codes = decoder_ring.keys()

    decoded = bytearray()
    buffer = ""

    # Iterate over each bit in the message, decoding and concatenating if a match is found
    for bit in message[8:-padding_length] if padding_length else message[8:]:
        buffer += bit

        if buffer in codes:
            decoded.append(decoder_ring[buffer])
            buffer = ""

    return bytes(decoded)


def compress(message: bytes) -> Tuple[array, Dict]:
    """Given the bytes read from a file, calls encode and turns the string into an array of bytes to be written to disk.

    :param message: raw sequence of bytes from a file
    :returns: array of bytes to be written to disk
              dict containing the decoder ring
    """
    encoded, decoder_ring = encode(message)
    return (
        int(encoded, 2).to_bytes(length=len(encoded) // 8, byteorder="big"),
        decoder_ring,
    )


def decompress(message: array, decoder_ring: Dict) -> bytes:
    """Given a decoder ring and an array of bytes read from a compressed file, turns the array into a string and calls decode.

    :param message: array of bytes read in from a compressed file
    :param decoder_ring: dict containing the decoder ring
    :return: raw sequence of bytes that represent a decompressed file
    """
    return decode("".join(map(lambda b: f"{b:08b}", message)), decoder_ring)


def count_words(message: bytes) -> dict[int, int]:
    """Count the frequency of each byte in a byte array.

    :param message: array of bytes
    :return: dict where the keys are the bytes (represented as integers 0-255) and the values are the number of occurrences
    """
    frequency = {}

    for byte in message:
        if byte not in frequency:
            frequency[byte] = 0

        frequency[byte] += 1

    return frequency


def build_tree(message: bytes) -> tuple[int, tuple]:
    """Build a Huffman tree from an array of bytes.

    :param message: array of bytes
    :return: the root node of the tree in the form `(frequency, (child_1, child_2))`
    """
    # Get word frequencies
    frequencies = count_words(message)

    # Create heap
    heap = []
    heapify(heap)

    # Add leaf nodes (words) to heap
    for value, count in frequencies.items():
        # Including unique ID prevents sorting errors when two nodes have the same count value
        heappush(heap, (count, id(value), value))

    # Invariant: The Huffman tree for the supplied message is a full binary search tree.
    # Initialization: The tree is only composed of leaf nodes (as many as there are unique words in the message).
    # Maintenance: During each iteration, the two least frequent nodes are branched together by a new, shared parent node.
    #   Every internal node thus has exactly two children.
    #   The less frequent of these nodes is always the left child.
    # Termination: The loop ends when all leaf nodes have been branched by an internal node.

    # Merge two smallest nodes and push back onto heap until only one node remains
    while len(heap) > 1:
        min1 = heappop(heap)
        min2 = heappop(heap)
        new_val = (min1[0] + min2[0], id(min1), (min1, min2))
        heappush(heap, new_val)

    return heappop(heap)


def calculate_codewords(
    left_child: tuple[int, int, int | tuple],
    right_child: tuple[int, int, int | tuple],
    _bits="",
    _codewords=None,
) -> dict[int, str]:
    """Recursively traverse a Huffman tree, assigning codewords to bytes based on their frequency
    (more frequent = shorter codeword).

    :param left_child: left child node
    :param right_child: right child node:
    :return: a dict mapping bytes to their codewords, both represented as ints
    """
    # Left traversal: append 0 to codeword
    # Right traversal: append 1 to codeword

    # Invariant: The most frequent words are given the shortest codewords.
    # Initialization: The codeword is empty, and traversal starts at the root node.
    # Maintenance: For each level of depth, one bit is added to the codeword.
    # Termination: When a leaf node is reached, it is assigned the current codeword.
    #   More frequent words are higher up in the tree, and so get shorter codewords.

    if _codewords is None:
        _codewords = {}

    _bits = _bits + "0"
    _, _, left_value = left_child

    # Base case: node is a leaf
    if isinstance(left_value, int):
        # Assign this node the current codeword
        _codewords[left_value] = _bits
    # Recursive case: node has children
    else:
        # Add '0' to bitstring
        # Recursively traverse child nodes
        calculate_codewords(*left_value, _bits=_bits, _codewords=_codewords)

    # Pop last bit off bitstring
    _bits = _bits[:-1]

    # Repeat process for right child node, adding '1' to the bitstring instead of '0'
    _bits = _bits + "1"
    _, _, right_value = right_child

    if isinstance(right_value, int):
        _codewords[right_value] = _bits
    else:
        calculate_codewords(*right_value, _bits=_bits, _codewords=_codewords)

    _bits = _bits[:-1]

    return _codewords


if __name__ == "__main__":
    usage = f"Usage: {sys.argv[0]} [ -c | -d | -v | -w ] infile outfile"
    if len(sys.argv) != 4:
        raise Exception(usage)

    operation = sys.argv[1]
    if operation not in {"-c", "-d", "-v", "-w"}:
        raise Exception(usage)

    infile, outfile = sys.argv[2], sys.argv[3]
    if not os.path.exists(infile):
        raise FileExistsError(f"{infile} does not exist.")

    if operation in {"-c", "-v"}:
        with open(infile, "rb") as fp:
            _message = fp.read()

        if operation == "-c":
            _message, _decoder_ring = compress(_message)
            with open(outfile, "wb") as fp:
                marshal.dump((pickle.dumps(_decoder_ring), _message), fp)
        else:
            _message, _decoder_ring = encode(_message)
            print(_message)
            with open(outfile, "wb") as fp:
                marshal.dump((pickle.dumps(_decoder_ring), _message), fp)

    else:
        with open(infile, "rb") as fp:
            pickleRick, _message = marshal.load(fp)
            _decoder_ring = pickle.loads(pickleRick)

        if operation == "-d":
            bytes_message = decompress(array("B", _message), _decoder_ring)
        else:
            bytes_message = decode(_message, _decoder_ring)
        with open(outfile, "wb") as fp:
            fp.write(bytes_message)
