/*
    Back-end audio utils.
*/

const fs = require('fs');

const wav = require('node-wav');

const miscUtils = require('../utils/misc_utils.js');
const constants = require('../constants_app.js');


function readWav(fpath) {
    // read wav file into Float32Array format
    // to get untyped array of floats for first channel: arr = Array.from(arr32.channelData[0])
    let buffer = fs.readFileSync(fpath);
    let result = wav.decode(buffer); // Float32Array
    return result;
}

function writeWav(fpath, wf) {
    // write Float32Array info to file
    console.log(`Writing ${wf.channelData[0].length} samples to ${fpath}`);
    let buffer = wav.encode(wf.channelData, { sampleRate: wf.sampleRate, float: true, bitDepth: 16 });
    fs.writeFileSync(fpath, buffer);
}

function getNextSeg(segsInfoAvail, tnow) {
    const audSegDur = miscUtils.readJSON().BUFF_DUR;
    let tend = new Date(tnow);
    tend.setMilliseconds(tend.getMilliseconds() + 1000 * audSegDur);
    for (segInfo of segsInfoAvail) {
        if (segInfo.datetime > tnow && segInfo.datetime < tend) {
            delete tend;
            return segInfo;
        }
    }
    delete tend;
    return {datetime: null, filepath: null, max_amp: null};
}

function getSegsInfo(segsInfoAvail, tmin, tmax) {
    // get all segment info in range
    const cjs = miscUtils.readJSON();
    const segDur = cjs.BUFF_DUR;

    let segsInfo = [];
    let tnow = new Date(tmin); // Date object
    while(tnow < tmax) {
        segsInfo.push(getNextSeg(segsInfoAvail, tnow));
        tnow.setMilliseconds(tnow.getMilliseconds() + 1000 * segDur);
    }

    // fill in missing segs if at least one segment was found
    if (!segsInfo.every(obj => obj.datetime == null)) {
        const idxNotNull = segsInfo.map(segInfo_ => segInfo_.datetime == null).indexOf(false);
        for (let i = 0; i < segsInfo.length; i++) {
            if (segsInfo[i].datetime == null) {
                let tnew = new Date(segsInfo[idxNotNull].datetime);
                tnew.setMilliseconds(tnew.getMilliseconds() + 1000 * (i - idxNotNull) * segDur);
                segsInfo[i].datetime = tnew;
            }
        }
    }

    return segsInfo;
}

function makeAudioMacroseg(segsInfoAvail, tmin, tmax, dt) {
    // get segments in range (filling zeros for missing segs)
    const segsInfo = getSegsInfo(segsInfoAvail, tmin, tmax);
    const segs = miscUtils.concatAudio(
        segsInfo.map(s => (s.filepath != null) ? readWav(s.filepath) : getEmptySeg())
    );

    // write to assets dir
    const macrosegFpath = constants.ASSETS_DIR + "macroseg_" + miscUtils.formatFEDt(dt) + ".wav";
    writeWav(macrosegFpath, segs);

    return {"macrosegFpath": macrosegFpath};
}

function getEmptySeg() {
    const cjs = miscUtils.readJSON();
    const segLen = cjs.BUFF_LEN;
    const sr = cjs.SAMPLERATE;

    return {
        "sampleRate": sr,
        "channelData": [new Float32Array(segLen)]
    };
}


module.exports = {
    makeAudioMacroseg,
    getSegsInfo
}
