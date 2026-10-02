"""MOTANAXY Synthetic Dataset Generator.

Procedurally generates simple Python algorithmic functions and their test cases.
This allows us to scale to 10k+ verified examples without needing external LLM APIs.
"""

import ast
import json
import random
from pathlib import Path


def random_name(prefix="func"):
    """Generate a random function name."""
    vowels = "aeiou"
    consonants = "bcdfghjklmnprstvwxyz"
    name = prefix + "_" + "".join(random.choice(consonants) + random.choice(vowels) for _ in range(2))
    return name


def generate_math_op():
    """Generate a simple math function."""
    ops = [
        ("return a + b", "add"),
        ("return a - b", "sub"),
        ("return a * b", "mul"),
        ("return a // b", "div"),
        ("return a ** b", "pow"),
        ("return a % b", "mod"),
        ("return a + b + c", "add3"),
        ("return (a + b) * c", "math1"),
        ("return a * b - c", "math2"),
    ]
    code_template, name_prefix = random.choice(ops)
    func_name = random_name(name_prefix)
    
    if "c" in code_template:
        params = "a, b, c"
        args_gen = lambda: (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
    else:
        params = "a, b"
        args_gen = lambda: (random.randint(1, 20), random.randint(1, 20))
        
    docstring = f"Perform basic math operation: {code_template.replace('return ', '')}."
    
    code = f"def {func_name}({params}):\n    return {code_template.replace('return ', '')}\n"
    return func_name, code, docstring, args_gen


def generate_list_op():
    """Generate a simple list manipulation function."""
    ops = [
        ("return max(lst)", "find_max", lambda: [random.randint(1, 100) for _ in range(5)]),
        ("return min(lst)", "find_min", lambda: [random.randint(1, 100) for _ in range(5)]),
        ("return sum(lst)", "sum_list", lambda: [random.randint(1, 20) for _ in range(4)]),
        ("return len(lst)", "get_length", lambda: [random.randint(1, 10) for _ in range(random.randint(1, 8))]),
        ("return lst[::-1]", "reverse", lambda: [random.randint(1, 50) for _ in range(3)]),
        ("return sorted(lst)", "sort_asc", lambda: [random.randint(1, 50) for _ in range(4)]),
        ("return lst[0] if lst else None", "get_first", lambda: [random.randint(1, 10)] * random.randint(1, 3)),
        ("return lst[-1] if lst else None", "get_last", lambda: [random.randint(1, 10)] * random.randint(1, 3)),
    ]
    code_template, name_prefix, arg_gen = random.choice(ops)
    func_name = random_name(name_prefix)
    
    docstring = f"Perform list operation: {name_prefix.replace('_', ' ')}."
    code = f"def {func_name}(lst):\n    {code_template}\n"
    
    return func_name, code, docstring, lambda: (arg_gen(),)


def build_example(generator_func):
    """Build a complete prompt/response pair by running the generated code."""
    func_name, code, docstring, args_gen = generator_func()
    
    # Compile the generated code securely
    local_env = {}
    try:
        exec(code, {}, local_env)
    except Exception:
        return None # Skip if bad syntax
        
    func = local_env[func_name]
    
    # Generate test cases
    test_cases = []
    for _ in range(3):
        args = args_gen()
        try:
            expected = func(*args)
            
            # Format args for code representation
            args_repr = ", ".join(repr(a) for a in args)
            test_cases.append(f"assert {func_name}({args_repr}) == {expected!r}")
        except Exception:
            continue
            
    if not test_cases:
        return None
        
    prompt = f"# {docstring}\n"
    full_text = prompt + code + "\n" + "\n".join(test_cases)
    
    return {
        "concept": func_name,
        "prompt": prompt,
        "code": code,
        "text": full_text
    }


def generate_dataset(num_examples=5000, out_file="training_data/synthetic_10k.jsonl"):
    """Generate thousands of verified examples."""
    generators = [generate_math_op, generate_list_op]
    
    results = []
    attempts = 0
    
    print(f"Generating {num_examples} verified examples...")
    while len(results) < num_examples and attempts < num_examples * 2:
        attempts += 1
        gen_func = random.choice(generators)
        example = build_example(gen_func)
        
        if example:
            results.append(example)
            
        if len(results) % 1000 == 0 and len(results) > 0:
            print(f"Generated {len(results)} examples...")
            
    out_path = Path(out_file)
    out_path.parent.mkdir(exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Done! Wrote {len(results)} verified examples to {out_file}")
    
    # Show a random example
    if results:
        print("\nSample generation:")
        print(random.choice(results)["text"])


if __name__ == "__main__":
    random.seed(42)
    # Generate 10,000 examples
    generate_dataset(10000)
