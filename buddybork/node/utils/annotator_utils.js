/*
    Back-end annotator utils.
*/

const db = require('../db/database_utils.js');
const constants = require('../constants_app.js');
const miscUtils = require('../utils/misc_utils.js');
const audUtils = require('../utils/audio_utils.js');


/* misc */
function getTimeRange(dt, extraPad=false) {
    // find time range for fetching segments
    const audSegDur = miscUtils.readJSON().BUFF_DUR;
    const halfwinDur = constants.ANNOT_SEG_HALFWIDTH * audSegDur;
    let winPad = 0.5 * audSegDur;
    if (extraPad) { winPad += 0.5 * audSegDur; }

    let tmin = new Date(dt);
    tmin.setMilliseconds(tmin.getMilliseconds() + 1000 * (- halfwinDur - winPad));
    const tmin_s = miscUtils.convertDateToString(tmin);

    let tmax = new Date(dt);
    tmax.setMilliseconds(tmax.getMilliseconds() + 1000 * (halfwinDur + winPad));
    const tmax_s = miscUtils.convertDateToString(tmax);

    return [tmin, tmax, tmin_s, tmax_s]
}


/* macroseg */
function macrosegQuery(tmin_s, tmax_s) {
    const condition = `WHERE datetime > '${tmin_s}' AND datetime < '${tmax_s}' ORDER BY datetime ASC`;
    return db.selectQuery("*", "raw", condition);
}

function makeMacroSeg(dt) {
    // make macroseg for specified datetime
    console.log(dt)

    const [tmin, tmax, tmin_s, tmax_s] = getTimeRange(dt);

    return macrosegQuery(tmin_s, tmax_s)
        .then(result => audUtils.makeAudioMacroseg(result, tmin, tmax, dt))
        .catch(err => console.log(err));
}

// annot tags
function getAnnotatorTags() {
    return db.selectQuery('*', 'tags', 'ORDER BY tag ASC')
        .then(result => {
            return {
                tags: result.map(row => miscUtils.strToOption(row.tag, true)),
                colors: miscUtils.getColors(result.length, 'jet', 0.8)
            }
        })
        .catch(err => console.log(err));
}

/* annot event */
function processAnnotEvent(tag, mode, dt, samp0, samp1=0) {
    const [tmin, tmax, tmin_s, tmax_s] = getTimeRange(dt);

    return macrosegQuery(tmin_s, tmax_s)
        .then(result => {
            const segsInfo = audUtils.getSegsInfo(result, tmin, tmax);
            if (segsInfo == null) { return null; } //  if no data in this macroseg, disallow annotation
            const dt0 = segsInfo[0].datetime;
            let dtAnnots = [samp0, samp1].map(s => getAnnotDtFromSamp(dt0, s));
            if (dtAnnots[0] > dtAnnots[1]) {
                dtAnnots = [dtAnnots[1], dtAnnots[0]];
            }
            switch (mode) {
                case "insert":
                    return processInsertAnnot(dtAnnots, tag);
                case "remove":
                    return processRemoveAnnot(dtAnnots[1], tag);
            }
        })
}

function getAnnotDtFromSamp(dt0, samp) {
    // dt0 is first dt in macroseg
    // samp is sample within macroseg
    const cjs = miscUtils.readJSON();
    const audSegDur = cjs.BUFF_DUR;
    const sr = cjs.SAMPLERATE;

    const dtAnnot = new Date(dt0);
    const adj = Math.round(1000 * (- audSegDur + samp / sr)); // time is captured at end of segment
    dtAnnot.setMilliseconds(dtAnnot.getMilliseconds() + adj);

    return dtAnnot
}

function getSampFromAnnotDt(dt0, dt) {
    // dt0 is first dt in macroseg
    // dt is annot datetime
    const cjs = miscUtils.readJSON();
    const audSegDur = cjs.BUFF_DUR;
    const sr = cjs.SAMPLERATE;

    const diff = dt.getTime() / 1000 - (dt0.getTime() / 1000 - audSegDur); // seconds
    const samp = Math.round(diff * sr);

    return samp
}

function processInsertAnnot(dts, tag) {
    const dts_s = dts.map(dt_ => miscUtils.convertDateToString(dt_));
    return db.insertQuery('annots', '(datetime_start, datetime_end, tag)', `('${dts_s[0]}', '${dts_s[1]}', '${tag}')`)
        .then(result => { return {"action": result}; });
}

function processRemoveAnnot(dt, tag) {
    let condition;

//    const dt_s = miscUtils.convertDateToString(dt);
//    condition = `WHERE datetime_start < '${dt_s}' AND datetime_end > '${dt_s}' AND tag = '${tag}'`;
    const dt_s = miscUtils.convertDateToString(dt);
    condition = `WHERE datetime_start = '${dt_s}' AND tag = '${tag}'`;
    return db.selectQuery('*', 'annots', condition)
        .then(result => {
            if (result.length != 1) {
                console.log('processRemoveAnnot() -> Processing first result, result.length = ' + result.length + ' != 1');
            }
            const result_ = result[0];
            const dts = [result_.datetime_start, result_.datetime_end];
            const dts_s = dts.map(dt_ => miscUtils.convertDateToString(dt_));
            condition = `datetime_start = '${dts_s[0]}' AND datetime_end = '${dts_s[1]}' AND tag = '${tag}'`;
            return db.deleteQuery('annots', condition)
                .then(result => { return {"action": result}; });
        })
}

async function getAnnotData(dt=null, day=null) {
    let condition = '';
    if (dt != null && day != null) { return }
    if (dt) {
        const [tmin, tmax, tmin_s, tmax_s] = getTimeRange(dt, true);
        condition = `WHERE datetime_start BETWEEN '${tmin_s}' AND '${tmax_s}'` +
            ` OR datetime_end BETWEEN '${tmin_s}' AND '${tmax_s}'`;
    }
    if (day) {
        condition = `WHERE date(datetime_start) = '${day}' OR date(datetime_end) = '${day}'`;
    }
    return db.selectQuery('*', 'annots', condition)
        .then(result => processAnnotData(result, dt))
}

async function processAnnotData(result, dt=null) {
    const annots = [];
    for (obj of result) { // process each annot one by one
        const annot = await processAnnotDataOne(obj, dt);
        annots.push(annot);
    }
    return annots
}

function processAnnotDataOne(obj, dt=null) {
    // convert annot info to object for return to client
    const annot = {
        datetime_start: miscUtils.convertDateToString(obj.datetime_start),
        datetime_end: miscUtils.convertDateToString(obj.datetime_end),
        tag: obj.tag
    }

    if (dt) {
        const [tmin, tmax, tmin_s, tmax_s] = getTimeRange(dt);

        return macrosegQuery(tmin_s, tmax_s)
            .then(result => {
                const dt0 = audUtils.getSegsInfo(result, tmin, tmax)[0].datetime; // get segment info for this macroseg
                const samp_start = getSampFromAnnotDt(dt0, obj.datetime_start); // get sample number for start dt
                const samp_end = getSampFromAnnotDt(dt0, obj.datetime_end); // " " end dt
                return {
                    ...annot,
                    samp_start: samp_start,
                    samp_end: samp_end
                }
            })
    }
    else {
        return annot
    }
}


function groupAnnotsByDay(annots) {
    annots_days = {};
    for (annot of annots) {
        const day = annot['datetime_start'].substring(0, 10);
        if (!Object.keys(annots_days).includes(day)) {
            annots_days[day] = [];
        }
        annots_days[day].push(annot);
    }
    return annots_days
}



module.exports = {
    makeMacroSeg,
    getAnnotatorTags,
    processAnnotEvent,
    getAnnotData,
    groupAnnotsByDay
}