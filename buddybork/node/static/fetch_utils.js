/*
    Front-end code for handling all requests (fetch operations) to the server.
*/

// fetch handler
function fetchHandler(route) {
//    console.log('Executing fetch at route: ' + route)
    return fetch(route, { method: 'POST' })
        .then(res => res.json())
}

// raw data routes
function getRawDataDayCounts()         { return fetchHandler('/raw/data/dayCounts'); }
function getRawDataForDay(day)         { return fetchHandler('/raw/data/timeline/' + day); } // day: YYYY-MM-DD
function getDataLims()                 { return fetchHandler('/raw/dtLims'); }
function getAllDataDays()              { return fetchHandler('/raw/allDays'); }

// annotator routes
function getAnnotDataAll()             { return fetchHandler('/annotator/data/annots'); }
function getAnnotDataAllGroupedByDay() { return fetchHandler('/annotator/data/annots/groupByDay'); }
function getAnnotDataForMacrosegPlot(macrosegDt_) {
    return fetchHandler('/annotator/data/annots/macroseg/' + macrosegDt_); // macrosegDt_: YYYY-MM-DD HH:mm:ss:SSS
}
function getAnnotDataForDay(day)       { return fetchHandler('/annotator/data/annots/day/' + day); }
function getTags()                     { return fetchHandler('/annotator/tags'); }
function getMacrosegData(macrosegDt_)  { return fetchHandler('/annotator/data/macroseg/' + macrosegDt_); }

// meta routes
function getMeta()                     { return fetchHandler('/meta'); }

// trainer routes
function fitModel(data)                { return fetchHandler('/trainer/model/fit/' + data); } // data: dict with info for fitting model
function predModel(data)               { return fetchHandler('/trainer/model/predict/' + data); }
function getModelInfo()                { return fetchHandler('/trainer/modelInfo'); }
function deleteModel(data)             { return fetchHandler('/trainer/model/delete/' + data); }

// detection routes
function getDetections(dt_s)           { return fetchHandler('/detector/get/' + dt_s); }
function setDetections(dt_s, modelId, dur='null') {
    return fetchHandler('/detector/set/' + dt_s + '/' + modelId + '/' + dur);
}
function setDataCaptureState(state)    { return fetchHandler('/dataCapture/set/' + state); } // state: bool string
function getDataCaptureState()         { return fetchHandler('/dataCapture/get'); }


// audio
async function fetchAudio(fpath) {
    // returns Promise that resolves Float32Array with waveform samples
    window.AudioContext = window.AudioContext || window.webkitAudioContext;
    const metaData = await getMeta();
    const audioContext = new AudioContext({
        sampleRate: metaData['SAMPLERATE']
    });

    return fetch(fpath)
        .then(response => response.arrayBuffer())
        .then(arrayBuffer => audioContext.decodeAudioData(arrayBuffer))
        .then(audioBuffer => audioBuffer.getChannelData(0))
}