#!/usr/bin/env python3
"""Quick script to verify Conv2d vs MaxPool dimension changes."""

print("=" * 70)
print("CONV2D DIMENSION FORMULA")
print("=" * 70)
print("Output Size = (Input Size - Kernel Size + 2*Padding) / Stride + 1")
print()

kernel = 3
padding = 1
stride = 1

print("For VGG-16 Conv2d layers:")
print(f"  Kernel Size = {kernel}")
print(f"  Padding = {padding}")
print(f"  Stride = {stride}")
print()

for input_size in [224, 112, 56, 28, 14]:
    output_size = (input_size - kernel + 2*padding) // stride + 1
    print(f"  Input: {input_size}x{input_size} -> Output: {output_size}x{output_size} (unchanged!)")

print()
print("=" * 70)
print("MAXPOOL2D DIMENSION FORMULA")
print("=" * 70)
print("Output Size = (Input Size - Kernel Size) / Stride + 1")
print()

kernel = 2
stride = 2

print("For VGG-16 MaxPool2d layers:")
print(f"  Kernel Size = {kernel}")
print(f"  Stride = {stride}")
print()

for input_size in [224, 112, 56, 28, 14]:
    output_size = (input_size - kernel) // stride + 1
    print(f"  Input: {input_size}x{input_size} -> Output: {output_size}x{output_size} (halved!)")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("✓ Conv2d: Changes channels (3->64->128->256->512), keeps spatial dims")
print("✓ MaxPool2d: Keeps channels, halves spatial dims (224->112->56->28->14->7)")
print()
print("VGG-16 Design Pattern:")
print("  [Conv + ReLU] x N -> [MaxPool] -> Repeat")
print("  - Convolutions extract features (increase channels)")
print("  - Pooling reduces spatial resolution (makes representation more abstract)")
