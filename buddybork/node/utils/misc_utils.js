/*
    Back-end misc utils.
*/

const fs = require('fs');
const colormaps = require('../other_modules/js-colormaps.js');

const timeChangeUtils = require('./time_change_utils.js');
const constants = require('../constants_app.js');


/* Functions */
function deepCopy(obj) {
    return JSON.parse(JSON.stringify(obj)); // returns a string for Date input
}

function convertDateToString(date) {
//const TZ_OFFSET = (timeChangeUtils.getTimeZoneOffset() * 60) * 60000; // time zone offset relative to UTC in ms
//const TZ_OFFSET = (new Date()).getTimezoneOffset() * 60000; // convert minutes to ms
    // applies time zone correction and removes GMT tags
    const tzoffset = timeChangeUtils.getTimeZoneOffset(date) * 60000;
    const newDate = new Date(date - tzoffset);
    const s = newDate.toISOString().replace('T', ' ').replace('Z', '');
//    delete newDate;
    return s;
}

function readJSON(path=null) {
    // read JSON from local file
    const path_ = (path) ? path : constants.CONST_JSON_PATH;
    return JSON.parse(fs.readFileSync(path_));
}

function writeJSON(path, obj) {
    // write object to JSON at specified path
    fs.writeFileSync(path, JSON.stringify(obj));
}

function initArray(n, v=0) {
    // initializes array of desired length with fill value
    let arr = [];
    arr.length = n;
    arr.fill(v);
    return arr
}

function initArrayRange(n, initVal=0) {
    let arr = [];
    for (let i = 0; i < n; i++) {
        arr.push(initVal + i);
    }
    return arr;
}

function concatTyped(resultConstructor, arrays) {
    const totalLength = arrays.reduce((total, arr) => {
        return total + arr.length
    }, 0);
    const result = new resultConstructor(totalLength);
    arrays.reduce((offset, arr) => {
        result.set(arr, offset);
        return offset + arr.length;
    }, 0);
    return result;
}

function concatAudio(arrays) {
    const wfs = arrays.map(arr => arr.channelData[0]);
    const data = {
        "sampleRate": arrays[0].sampleRate,
        "channelData": [ concatTyped(Float32Array, wfs) ]
    }
    return data;
}

function strToOption(s, labelUnderscoreToSpace) {
    const obj = {"label": s, "value": s};
    if (labelUnderscoreToSpace) {
        obj.label = obj.label.split('_').join(' ');
    }
    return obj;
}

function getDailyDtCounts(dts) {
    const days = dts.map(dt => splitDt(convertDateToString(dt))[0]);
    const daysUnique = [...new Set(days)];
    const counts = {};
    daysUnique.forEach(day => { counts[day] = 0; });
    days.forEach(day => { counts[day] += 1; });
    return counts;
}

function splitDt(dt) {
    // output: [date string, time string]
    return dt.split(' ');
}

function getColors(numColors, cmap='jet', mult=1) {
    return initArrayRange(numColors).map(i => {
        const color = colormaps.evaluate_cmap(i / (numColors - 1), cmap, false);
        return (mult == 1) ? color : applyColorMult(color, mult);
    });
}

function applyColorMult(color, mult) {
    return color.map(c => c * mult);
}

function setDifference(setA, setB) {
  const _difference = new Set(setA);
  for (const elem of setB) {
    _difference.delete(elem);
  }
  return _difference;
}

function arrDifference(arrA, arrB) {
    // same as setDifference but for arrays
    return Array.from(setDifference(new Set(arrA), new Set(arrB)));
}

function formatBEDt(dt) {
    return dt.replace('_', ' ');
}

function formatFEDt(dt) {
    return dt.replace(' ', '_');
}

function hashString(s) {
    const numChars = constants.HASH_CHARS.length;
    const charIdxs = [];
    for (c of s) {
        charIdxs.push(constants.HASH_CHARS.indexOf(c));
    }
    let hash = '';
    for (let i = 0; i < constants.LEN_HASH; i++) {
        const j = (13 * (i + 1) * charIdxs[i % s.length] + 17) % numChars;
        hash += constants.HASH_CHARS[j];
    }
    return hash;
}

function isNumeric(num) {
    return !isNaN(num);
}

function setEquality(xs, ys) {
    return (xs.size === ys.size) && ([...xs].every((x) => ys.has(x)));
}

function arrEquality(xs, ys) {
    return setEquality(new Set(xs), new Set(ys));
}



module.exports = {
    deepCopy,
    convertDateToString,
    readJSON,
    writeJSON,
    initArray,
    initArrayRange,
    concatTyped,
    concatAudio,
    strToOption,
    getDailyDtCounts,
    splitDt,
    getColors,
    setDifference,
    arrDifference,
    formatBEDt,
    formatFEDt,
    hashString,
    isNumeric,
    setEquality,
    arrEquality
};

