import tkinter as tk
from tkinter import messagebox
import nltk
from nltk.corpus import words

# ------------------------------------------------
# 1) DOWNLOAD WORDS (only once)
# ------------------------------------------------
nltk.download("words")
word_list = set(w.lower() for w in words.words())  # For dictionary matching

# ------------------------------------------------
# 2) ENCRYPTION FUNCTIONS
#    (same as before, for demonstration)
# ------------------------------------------------

def position_shift_encrypt(text):
    """Encrypt using position shift: each char shifted by (i+1)."""
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    encrypted = []

    for i, char in enumerate(text):
        if char.isalpha():
            is_upper = char.isupper()
            base_char = char.lower()
            old_index = alphabet.index(base_char)
            new_index = (old_index + (i + 1)) % 26
            new_char = alphabet[new_index]
            encrypted.append(new_char.upper() if is_upper else new_char)
        else:
            encrypted.append(char)
    return "".join(encrypted)

def caesar_encrypt(text, shift=1):
    """Standard Caesar cipher with a fixed shift."""
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    encrypted = []

    for char in text:
        if char.isalpha():
            is_upper = char.isupper()
            base_char = char.lower()
            old_index = alphabet.index(base_char)
            new_index = (old_index + shift) % 26
            new_char = alphabet[new_index]
            encrypted.append(new_char.upper() if is_upper else new_char)
        else:
            encrypted.append(char)
    return "".join(encrypted)

# ------------------------------------------------
# 3) DECRYPTION FUNCTIONS
# ------------------------------------------------

def position_shift_decrypt(cipher_text):
    """
    Decrypts a position-shift-encrypted message.
    Reverses the shift: subtract (i+1) from each character's index.
    Returns lowercase for easier dictionary checking.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    decrypted_chars = []

    for i, char in enumerate(cipher_text.lower()):
        if char in alphabet:
            old_index = alphabet.index(char)
            new_index = (old_index - (i + 1)) % 26
            decrypted_chars.append(alphabet[new_index])
        else:
            decrypted_chars.append(char)
    return "".join(decrypted_chars)

def caesar_decrypt_with_shift(cipher_text, shift):
    """
    Decrypts assuming a given Caesar shift.
    Returns lowercase for easier dictionary checking.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    decrypted_chars = []

    for char in cipher_text.lower():
        if char in alphabet:
            old_index = alphabet.index(char)
            new_index = (old_index - shift) % 26
            decrypted_chars.append(alphabet[new_index])
        else:
            decrypted_chars.append(char)
    return "".join(decrypted_chars)

# ------------------------------------------------
# 4) AUTO-DETECTION: Position Shift vs Caesar
# ------------------------------------------------

def auto_decrypt_both(cipher_text):
    """
    1) Try position-shift decryption.
    2) Try all standard Caesar shifts (1..25).
    3) Compare valid-word counts, pick the best method.
    Returns:
      (best_method, best_shift, best_decryption)
    Where best_shift is None for position-shift,
    or the integer shift for Caesar.
    """

    # --- 1) Attempt position-shift
    pos_decrypted = position_shift_decrypt(cipher_text)  # all lower
    pos_words = pos_decrypted.split()
    pos_valid_count = sum(1 for w in pos_words if w in word_list)

    # --- 2) Attempt Caesar for all shifts
    best_caesar_text = ""
    best_caesar_shift = None
    best_caesar_valid_count = 0

    for shift in range(1, 26):
        candidate = caesar_decrypt_with_shift(cipher_text, shift)
        candidate_words = candidate.split()
        valid_count = sum(1 for w in candidate_words if w in word_list)

        if valid_count > best_caesar_valid_count:
            best_caesar_valid_count = valid_count
            best_caesar_text = candidate
            best_caesar_shift = shift

    # --- 3) Compare results
    if pos_valid_count == 0 and best_caesar_valid_count == 0:
        # No luck on either method
        return ("Unknown", None, "Could not determine decryption (no valid words).")

    if pos_valid_count >= best_caesar_valid_count:
        # Position-shift wins or ties
        return ("Position Shift", None, pos_decrypted)
    else:
        # Caesar wins
        return ("Caesar", best_caesar_shift, best_caesar_text)

# ------------------------------------------------
# 5) TKINTER GUI
# ------------------------------------------------

def encrypt_message():
    text = input_text.get()
    method = encryption_method.get()

    if not text:
        messagebox.showwarning("Input Error", "Please enter a message to encrypt!")
        return

    if method == "Position Shift":
        encrypted = position_shift_encrypt(text)
    else:
        # Standard Caesar with shift=1
        encrypted = caesar_encrypt(text, shift=1)

    output_text.set(encrypted)

def decrypt_message():
    text = input_text.get()

    if not text:
        messagebox.showwarning("Input Error", "Please enter a message to decrypt!")
        return

    # AUTO-DETECT BOTH
    best_method, best_shift, best_decryption = auto_decrypt_both(text)

    if best_method == "Unknown":
        output_text.set(best_decryption)  # "Could not determine..."
    elif best_method == "Position Shift":
        output_text.set(f"Detected: Position Shift\n\nDecrypted:\n{best_decryption}")
    else:
        # best_method == "Caesar"
        output_text.set(
            f"Detected: Standard Caesar (shift={best_shift})\n\nDecrypted:\n{best_decryption}"
        )

# GUI Setup
root = tk.Tk()
root.title("Cipher Auto-Decrypt (Position vs Caesar)")
root.geometry("480x320")
root.resizable(False, False)

# Input Label + Entry
tk.Label(root, text="Enter Message:").pack(pady=5)
input_text = tk.StringVar()
tk.Entry(root, textvariable=input_text, width=50).pack()

# Encryption Method (for demonstration)
encryption_method = tk.StringVar(value="Position Shift")
tk.Label(root, text="Select Encryption Method (for Encrypt):").pack(pady=5)
tk.Radiobutton(root, text="Position Shift", variable=encryption_method, value="Position Shift").pack()
tk.Radiobutton(root, text="Caesar (Shift=1)", variable=encryption_method, value="Caesar").pack()

# Encrypt Button
tk.Button(root, text="Encrypt", command=encrypt_message, width=15).pack(pady=5)

# Decrypt Button
tk.Button(root, text="Auto-Decrypt", command=decrypt_message, width=20).pack(pady=5)

# Output
tk.Label(root, text="Output:").pack(pady=5)
output_text = tk.StringVar()
output_label = tk.Label(root, textvariable=output_text, fg="blue", font=("Arial", 11, "bold"), wraplength=450, justify="left")
output_label.pack()

root.mainloop()
