// Bitvector Operations with OxiZ WASM
// Run with: node --experimental-wasm-modules wasm_bitvector.mjs
//
// This example demonstrates bitvector operations, useful for:
// - Hardware verification
// - Low-level programming verification
// - Cryptographic property checking

import init, { WasmSolver } from 'oxiz-wasm';

async function main() {
    await init();
    const solver = new WasmSolver();

    console.log("=== Bitvector Operations Example ===\n");

    // Use QF_BV logic (Quantifier-Free Bitvectors)
    solver.setLogic("QF_BV");

    // Example 1: Find inputs for bitwise AND
    console.log("--- Example 1: Bitwise AND ---");
    solver.push();

    solver.declareConst("x", "(_ BitVec 8)");
    solver.declareConst("y", "(_ BitVec 8)");
    solver.declareConst("result", "(_ BitVec 8)");

    // result = x AND y
    solver.assertFormula("(= (bvand x y) result)");
    // We want result to be 0x0F
    solver.assertFormula("(= result #x0F)");
    // x must have all upper bits set
    solver.assertFormula("(bvuge x #xF0)");

    if (solver.checkSat() === "sat") {
        const model = solver.getModel();
        console.log("Found: x =", model.x.value, ", y =", model.y.value);
        console.log("x AND y =", model.result.value);
    }
    solver.pop();
    solver.resetAssertions();

    // Example 2: Overflow detection
    console.log("\n--- Example 2: Addition Overflow ---");
    solver.push();

    solver.declareConst("a", "(_ BitVec 8)");
    solver.declareConst("b", "(_ BitVec 8)");
    solver.declareConst("sum", "(_ BitVec 8)");

    // sum = a + b (with wrap-around)
    solver.assertFormula("(= (bvadd a b) sum)");
    // Find case where overflow occurs (sum < a and sum < b)
    solver.assertFormula("(bvult sum a)");
    solver.assertFormula("(bvugt a #x80)"); // a > 128

    if (solver.checkSat() === "sat") {
        const model = solver.getModel();
        console.log("Overflow example found:");
        console.log("  a =", model.a.value);
        console.log("  b =", model.b.value);
        console.log("  a + b (8-bit) =", model.sum.value);
    }
    solver.pop();
    solver.resetAssertions();

    // Example 3: Find x where x XOR x = 0
    console.log("\n--- Example 3: XOR Self-Inverse ---");
    solver.push();

    solver.declareConst("val", "(_ BitVec 16)");
    solver.assertFormula("(= (bvxor val val) #x0000)");
    solver.assertFormula("(distinct val #x0000)"); // val != 0

    if (solver.checkSat() === "sat") {
        const model = solver.getModel();
        console.log("Found val =", model.val.value);
        console.log("val XOR val = 0x0000 (as expected for any value)");
    }
    solver.pop();
    solver.resetAssertions();

    // Example 4: Bit manipulation puzzle
    console.log("\n--- Example 4: Set specific bit ---");
    solver.push();

    solver.declareConst("input", "(_ BitVec 8)");
    solver.declareConst("output", "(_ BitVec 8)");
    solver.declareConst("mask", "(_ BitVec 8)");

    // output = input OR mask (set bits where mask is 1)
    solver.assertFormula("(= (bvor input mask) output)");
    // We want bit 3 to be set in output
    solver.assertFormula("(= ((_ extract 3 3) output) #b1)");
    // Bit 3 is NOT set in input
    solver.assertFormula("(= ((_ extract 3 3) input) #b0)");
    // Mask should be minimal (only bit 3 set)
    solver.assertFormula("(= mask #x08)");

    if (solver.checkSat() === "sat") {
        const model = solver.getModel();
        console.log("Input:", model.input.value);
        console.log("Mask:", model.mask.value);
        console.log("Output:", model.output.value);
    }
    solver.pop();

    solver.free();
    console.log("\nDone!");
}

main().catch(console.error);
