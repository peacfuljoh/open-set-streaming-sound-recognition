/* Miscellaneous utility functions for front-end */

// EST or EDT
function getTimeChangeStatus(dt) {
    // beginning of EDT
    const dateEDTStart = new Date(dt);
    dateEDTStart.setMonth(2, 12); // month, day (March 12)
    dateEDTStart.setHours(3, 0, 0, 0); // hour, minute, second, ms (7 AM) --> this set op depends on the current time zone!!!

    // end of EDT
    const dateEDTEnd = new Date(dt);
    dateEDTEnd.setMonth(10, 5); // month, day (Nov 5)
    dateEDTEnd.setHours(2, 0, 0, 0); // hour, minute, second, ms (6 AM) --> this set op depends on the current time zone!!!

    // determine if EST or EDT
    return (dt >= dateEDTStart && dt <= dateEDTEnd) ? 'EDT' : 'EST';
}

// negated offset between UTC and EST/EDT in minutes
function getTimeZoneOffset(dt) {
    return dt.getTimezoneOffset(); // uses local time as reference
//    return getTimeChangeStatus(dt) == 'EST' ? 5 : 4;
}

function deepCopy(obj) {
    return JSON.parse(JSON.stringify(obj)); // returns a string for Date input
}

function convertDateToString(date) {
    // applies time zone correction and removes GMT tags
//    const tzoffset = (getTimeZoneOffset() * 60) * 60000;
    const tzoffset = getTimeZoneOffset(date) * 60000;
    const newDate = new Date(date - tzoffset);
    const s = newDate.toISOString().replace('T', ' ').replace('Z', '');
    delete newDate;
    return s;
}

// misc
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

function getDateRange(dt) {
    // get datetime endpoints for a datetime's day
    let t0 = new Date(dt);
    t0.setHours(0, 0, 0);
    let t1 = new Date(dt);
    t1.setDate(t1.getDate() + 1);
    t1.setHours(0, 0, 0);
    return [t0, t1];
}

function setDropdownOptions(id_, opts) {
    const elem = getElem(id_);
    elem.options.length = opts.length;
    for (let i = 0; i < opts.length; i++) {
        elem.options[i] = new Option(opts[i].label, opts[i].value);
    }
    setDisplay(id_, 'block');
}

function setDisplay(id_, mode) {
    const elem = getElem(id_);
    elem.style.display = mode;
}

function getDisplay(id_) {
    return getElem(id_).style.display;
}

function maxAbsArray(arr) {
    return arr.reduce((acc, val) => Math.max(acc, Math.abs(val)), 0);
}

function strToOption(s) {
    return {"label": s, "value": s};
}

function splitDt(dt) {
    // output: [date string, time string]
    return dt.split(' ');
}

function getElem(id_) {
    return document.getElementById(id_)
}

function padDtLims(dtLims, numDays=7) {
    // pad upper and lower datetime limits
    let dt;

    for (let i = 0; i < 2; i++) {
        dt = new Date(dtLims[i]);
        dt.setDate(dt.getDate() + (i == 0 ? -numDays : numDays));
        dtLims[i] = dt;
    }

    delete dt;
    return dtLims;
}

// generate random permutation
function makePermutation(N) {
	let idxs = Array.from(Array(N).keys());
	let perm = Array(N);
	for (let i = 0; i < N; i++) {
		let j = Math.floor(Math.random() * (N - i));
		perm[i] = idxs[j];
		idxs.splice(j, 1);
	}
	return perm;
}

// permute an array
function permuteArray(arr) {
	const N = arr.length;
	let perm = makePermutation(N);
	let arr_perm = Array(N);
	for (let i = 0; i < N; i++) {
		arr_perm[i] = arr[perm[i]];
	}
	return arr_perm;
}

// generate deterministic permutation
function makePermutationDet(N, step=7) {
	let idxs = initArrayRange(N);
	let perm = Array(N);
	for (let i = 0; i < N; i++) {
		let j = (i * step) % idxs.length;
		perm[i] = idxs[j];
		idxs.splice(j, 1);
	}
	return perm;
}

function formatFEDt(dt) {
    return dt.replace(' ', '_');
}

function formatBEDt(dt) {
    return dt.replace('_', ' ');
}

function getDtNowString() {
    return convertDateToString(new Date());
}

function ampToDb(amp, maxAmp=null, minAmp=1e-3) {
    const maxAmp_ = (maxAmp == null) ? Math.pow(2, 15) : maxAmp;
    return 20 * Math.log10(amp / maxAmp_ + minAmp)
}

// crop objects of different types by min and max days
function cropByDayRange(obj, type, dayMin, dayMax) {
    if (type == 'arrOfAnnots') {
        obj = obj.filter(x => (splitDt(x['datetime_start'])[0] <= dayMax))
    }
    if (type == 'obj') { // object with day strings as keys
        for (day in obj) {
            if (day > dayMax) { delete obj[day]; }
        }
    }
    if (type == 'arrOfDays') { // array with day strings
        obj = obj.filter(x => (x <= dayMax));
    }
    if (type == 'dtLims') { // 2-element array with min and max datetime strings
        obj[1] = dayMax + ' 00:00:00.000';
    }
    return obj;
}