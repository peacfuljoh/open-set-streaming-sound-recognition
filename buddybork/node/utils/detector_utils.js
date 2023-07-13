/*
    Back-end detection utils.
*/

const db = require('../db/database_utils.js');
const miscUtils = require('../utils/misc_utils.js');
const pyUtils = require('../utils/py_utils.js');
const constants = require('../constants_app.js');



function getDetectionWin(dt, dur=null) {
    const winDur = (dur == null) ? constants.DETECTION_DUR : dur;

    const tmin = new Date(dt);
    tmin.setMinutes(tmin.getMinutes() - winDur);
    const tmin_s = miscUtils.convertDateToString(tmin);

    const tmax = new Date(dt);
    const tmax_s = miscUtils.convertDateToString(tmax);

    return [tmin, tmax, tmin_s, tmax_s];
}

function getMaxAmps(dts) {
    // dts is array of non-GMT datetime strings, need not be sorted
    if (dts.length == 0) { return []; }
    const dtsSorted = miscUtils.deepCopy(dts).sort();
    const tmin_s = dtsSorted[0];
    const tmax_s = dtsSorted[dtsSorted.length - 1];
    const condition = `WHERE datetime >= '${tmin_s}' AND datetime <= '${tmax_s}' ORDER BY datetime ASC`;
    return db.selectQuery("*", "raw", condition)
        .then(results => {
            const dtsQ = results.map(row => miscUtils.convertDateToString(row.datetime));
            const idxs = dts.map(dt_ => dtsQ.indexOf(dt_));
            const maxAmps = idxs.map(idx => {
                return (idx == -1) ? 1e-3 : results[idx].max_amp;
            });
            return maxAmps;
        })
}

async function getDetections(dt) {
    // get detection info (datetime and tag)
    const [tmin, tmax, tmin_s, tmax_s] = getDetectionWin(dt);
    const condition = `WHERE datetime > '${tmin_s}' AND datetime < '${tmax_s}' ORDER BY datetime ASC`;
    const results = await db.selectQuery("*", "detections", condition);
    const numRows = results.length;

    // get max amps and build output
    const dts = results.map(row => miscUtils.convertDateToString(row.datetime));
    const maxAmps = await getMaxAmps(dts);

    return miscUtils.initArrayRange(numRows).map(i => {
        return {
            datetime: dts[i],
            tag: results[i].tag,
            max_amp: maxAmps[i]
        }
    })
}

async function setDetections(dt, modelId, detDur=null) {
    /* Set detections by running all segs within range of specified datetime through specified model */
    // get raw seg and detection info in datetime range
    const [tmin, tmax, tmin_s, tmax_s] = getDetectionWin(dt, detDur);

    const condition = `WHERE datetime > '${tmin_s}' AND datetime < '${tmax_s}'`;
    const rawRows = await db.selectQuery('*', 'raw', condition);
    const detRows = await db.selectQuery('*', 'detections', condition);

    // identify segs without detections
    const dtRaw = rawRows.map(row => miscUtils.convertDateToString(row.datetime));
    const dtDet = detRows.map(row => miscUtils.convertDateToString(row.datetime));
    const dtsToProcess = miscUtils.arrDifference(dtRaw, dtDet);
//    let dtsToProcess = Array.from(miscUtils.setDifference(new Set(dtRaw), new Set(dtDet)));
    const numDts = dtsToProcess.length;

    // perform detection on un-tagged segs
    const numBatches = Math.ceil(numDts / constants.DET_BATCH_SIZE)
    for (let i = 0; i < numBatches; i++) {
        const dts_ = dtsToProcess.slice(i * constants.DET_BATCH_SIZE, Math.min(numDts, (i + 1) * constants.DET_BATCH_SIZE));
        console.log(`setDetections() -> Running batch ` + (i + 1) + ` of ` + numBatches + `, size ${dts_.length}.`)
        const params = JSON.stringify({dts: dts_, model_id: modelId});
        await pyUtils.runPyOp(constants.PYDETECTION_PATH, ['detect', params])
            .then(obj => insertDetections(dts_, obj.tags))
    }

    return null;
}

function insertDetections(dtsToProcess, tagsPred) {
    // insert new detections into database
    const numDts = dtsToProcess.length;
    let values = '';
    for (let i = 0; i < numDts; i++) {
        values += `('${dtsToProcess[i]}', '${tagsPred[i]}')`;
        if (i < numDts - 1) { values += `,`; }
    }
    return db.insertQuery('detections', '(datetime, tag)', values)
}

module.exports = {
    getDetections,
    setDetections
}
