"""A tiny end-to-end loop using several tools together:

 E  retrieve similar solved cases from memory        (CaseLibrary)
 F  predict -> act -> observe -> revise              (RevisionLoop)
 D  synthesize a program from examples                (EnumerativeSynthesizer)
 G  check it against unit tests                       (UnitTests, restricted Python evaluator)
 E  store the verified solution back in memory

Use backend="python-subprocess" to run candidates in a resource-limited child process instead.

"""
import saturncube as sc


def main():
    lib = sc.CaseLibrary()
    lib.add("absolute value of an integer", "def f(x):\n    return -x if x < 0 else x\n", {"conditional"}, True, True)
    lib.add("sum of squares of two integers", "def f(x, y):\n    return x * x + y * y\n", {"arithmetic"}, True, True)

    request = "return the larger of two integers"
    examples = [((a, b), max(a, b)) for a in range(-3, 4) for b in range(-3, 4)]
    tests = sc.UnitTests("f", examples)                    # backend="python" by default

    def propose(task, attempts):
        if not attempts:                                   # first try: adapt the closest remembered case
            return lib.retrieve(task, k=1, concepts={"conditional"})[0].code
        synth = sc.EnumerativeSynthesizer(["x", "y"], max_size=7)   # then: synthesize from the examples
        return synth.to_function(synth.synthesize(examples))

    def verify(src):
        rep = tests.run(src)
        return rep.ok, f"{rep.passed}/{rep.total} tests passed" + ("" if rep.compile_ok else f"; invalid code: {rep.message[:80]}")

    loop = sc.RevisionLoop(propose, verify, max_rounds=3)
    solution = loop.run(request)
    for a in loop.attempts_:
        print(f"round {a.round}: {'OK ' if a.ok else 'FAIL'} {a.feedback}")
    if solution:
        lib.add(request, solution, {"conditional", "comparison"}, True, True)
        print("verified solution stored; library now has", len(lib), "cases:\n" + solution)
    return solution


if __name__ == "__main__":
    main()
