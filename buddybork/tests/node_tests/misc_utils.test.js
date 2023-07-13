
const misc_utils = require('../../node/utils/misc_utils.js');


describe("misc_utils helper methods", () => {
//    test("it should convert a date to a string", () => {
//        const input = new Date('2023-04-22T22:54:30.971Z');
//        const output = '2023-04-22 18:54:30.971';
//        expect(misc_utils.convertDateToString(input)).toEqual(output);
//    });
    test("it should initialize an array filled with a constant value", () => {
        const n = 3;
        const v = 1;
        const output = [v, v, v];
        expect(misc_utils.initArray(n, v)).toEqual(output);
    });
    test("it should initialize an array with a range of values", () => {
        const n = 3;
        const initVal = 5;
        const output = [5, 6, 7];
        expect(misc_utils.initArrayRange(n, initVal)).toEqual(output);
    });
    test("it should concatenate arrays of a non-default type", () => {
        const arrays = [new Float32Array([0, 1, 2]), new Float32Array([3, 4, 5])];
        const resultConstructor = Float32Array;
        const output = new Float32Array([0, 1, 2, 3, 4, 5]);
        expect(misc_utils.concatTyped(resultConstructor, arrays)).toEqual(output);
    });
    test("it should create an options object from a string", () => {
        const s = 'abc'
        const labelUnderscoreToSpace = true;
        const output = {"label": s, "value": s};
        expect(misc_utils.strToOption(s, labelUnderscoreToSpace)).toEqual(output);
    });
//    test("it should list counts of datetimes binned by day", () => {
//        const input = [
//            new Date('2023-04-22T22:54:30.971Z'),
//            new Date('2023-04-22T22:54:30.971Z'),
//            new Date('2023-04-03T22:54:30.971Z')
//        ];
//        // ...
//        expect(misc_utils.getDailyDtCounts(dts)).toEqual(output);
//    });
    test("it should split a string by the space character", () => {
        const input = '2023-04-22 18:54:30.971';
        const output = ['2023-04-22', '18:54:30.971'];
        expect(misc_utils.splitDt(input)).toEqual(output);
    });
    test("it should return the logical difference between two sets", () => {
        const setA = new Set([0, 1, 2, 3, 4, 5]);
        const setB = new Set([2, 4]);
        const output = new Set([0, 1, 3, 5]);
        expect(misc_utils.setDifference(setA, setB)).toEqual(output);
    });
    test("it should return the logical difference between two sets (inputs/outputs are arrays)", () => {
        const setA = [0, 1, 2, 3, 4, 5];
        const setB = [2, 4];
        const output = [0, 1, 3, 5];
        expect(misc_utils.arrDifference(setA, setB)).toEqual(output);
    });
    test("it should replace _ with a space in a string", () => {
        const input = '2023-04-22_18:54:30.971';
        const output = '2023-04-22 18:54:30.971';
        expect(misc_utils.formatBEDt(input)).toEqual(output);
    });
    test("it should replace a space with _ in a string", () => {
        const input = '2023-04-22 18:54:30.971';
        const output = '2023-04-22_18:54:30.971';
        expect(misc_utils.formatFEDt(input)).toEqual(output);
    });
    test("it should create a hash string from an input string", () => {
        const input = 'a8b6d4m0c8264nxl6gp';
        const output = 'r$4j&$DzJPJ$XhMbt1Yr1phrbP1zD1';
        expect(misc_utils.hashString(input)).toEqual(output);
    });
});
