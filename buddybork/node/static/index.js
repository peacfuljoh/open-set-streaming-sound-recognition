/*
    Front-end code for handling interaction on main Dashboard page, e.g. detection, toggle stream ON/OFF.
*/

detTimelinePlotMode = 'tag'; // 'amp', 'tag'

const detIntvlDur = 1; // minutes

const audioElem = getElem('audioPlayback');



/* STARTUP */
// model select menu
setupModelSelectMenu()
    .then(runDetFunc(detIntvlDur))
    .then(setInterval(function() { runDetFunc(detIntvlDur); }, 1000 * 60 * detIntvlDur))

// data capture switch (ON/OFF)
getDataCaptureState()
    .then(obj => getElem('dataCollectCheckbox').checked = obj.capture_state);



/* MISC */
function mergeTags(tags1, tags2) {
    return Array.from(new Set([...tags1, ...tags2])).sort();
}


/* SETUP */
function setupModelSelectMenu() {
    // model select
    return getModelInfo()
        .then(modelInfo => {
            const modelIds = Object.keys(modelInfo);
            let dropdownOptions = [{label: '', value: ''}];
            const modelIdOptions = modelIds.map(id_ => {
                return {label: modelInfo[id_].model_name, value: id_};
            });
            dropdownOptions = dropdownOptions.concat(modelIdOptions);
            setDropdownOptions("modelSelectDropdown", dropdownOptions);
            getElem('modelSelectDropdown').value = dropdownOptions[0].value;
        });
}


/* PLOTS */
// detection plot
async function updateDetTimeline() {
    const dtNow = getDtNowString();
    const tagsInfo = await getTags(); // keys: tags, colors
    const annots = await getDetections(formatFEDt(dtNow));

    drawDetTimeline(annots, tagsInfo, dtNow);
    setTimelineEvents();
}

function drawDetTimeline(annots, tagsInfo, dtNow) {
    // get consolidated tag info
    const [tagsOrig, tagColorsOrig] = getTagsAndColorsFromTagsInfo(tagsInfo);
    const annotsTags = annots.map(annot => annot.tag);
    const tags = mergeTags(tagsOrig, annotsTags);
    const tagColors = [];
    for (tag of tags) {
        const idx = tagsOrig.indexOf(tag);
        if (idx == -1) { tagColors.push("rgba(0, 0, 0, 1)"); }
        else           { tagColors.push(tagColorsOrig[idx]); }
    }

    // pack info for plot
    if (detTimelinePlotMode == 'amp')      { func = prepDetTimelineByAmp; }
    else if (detTimelinePlotMode == 'tag') { func = prepDetTimelineByTags; }
    const [dataAnnots, chartLayout_] = func(annots, tags, tagColors);

    // get start and end times for plot
    let dtStart = new Date(dtNow);
    dtStart.setMinutes(dtStart.getMinutes() - 6 * 60 * 1.02);
    dtStart = convertDateToString(dtStart);

    let dtEnd = new Date(dtNow);
    dtEnd.setMinutes(dtEnd.getMinutes() + 6 * 60 * 0.02);
    dtEnd = convertDateToString(dtEnd);

    // update plot date range
    chartLayout_.xaxis.range = [dtStart, dtEnd];
    if (detTimelinePlotMode == 'amp') { chartLayout_.yaxis.range = [-50, 0]; }
    chartLayout_.title.text = "Detections";
    chartLayout_.xaxis.title.text = "Time";

    // draw
    updatePlot("detectionsDiv", dataAnnots, chartLayout_);
}

function prepDetTimelineByTags(annots, tags, tagColors) {
    // plot data
    const dataAnnots = [];
    const tagCounts = {};
    for (tag of tags) {
        const tagIdx = tags.indexOf(tag);
        const annotIdxs = [];
        annots.forEach((annot, idx) => { (annot.tag == tag) ? annotIdxs.push(idx) : null; });
        tagCounts[tag] = annotIdxs.length;
        dataAnnots.push({
            x: annotIdxs.map(idx => annots[idx].datetime),
            y: initArray(tagCounts[tag], tagIdx),
            type: 'scatter',
            marker: { color: tagColors[tagIdx], size: 5 },
            mode: 'markers',
            name: ''
        })
    }

    // plot meta
    const chartLayout_ = deepCopy(chartLayout);
    chartLayout_.yaxis.tickvals = initArrayRange(tags.length);
    chartLayout_.yaxis.ticktext = tags.map(tag => tag + ' (' + tagCounts[tag] + ')');
    chartLayout_.yaxis.tickfont = {size: 10};
    chartLayout_.yaxis.range = [-0.5, tags.length];
    chartLayout_.showlegend = false;

    return [dataAnnots, chartLayout_]
}

function prepDetTimelineByAmp(annots, tags, tagColors) {
    // plot data
    const dataAnnots = [];
    const tagCounts = {};
    for (tag of tags) {
        const tagIdx = tags.indexOf(tag);
        const annotIdxs = [];
        annots.forEach((annot, idx) => { (annot.tag == tag) ? annotIdxs.push(idx) : null; });
        tagCounts[tag] = annotIdxs.length;
        if (tagCounts[tag] > 0) {
            dataAnnots.push({
                x: annotIdxs.map(idx => annots[idx].datetime),
                y: annotIdxs.map(idx => ampToDb(annots[idx].max_amp)),
                type: 'scatter',
                marker: { color: tagColors[tagIdx], size: 5 },
                mode: 'markers',
                name: tag + ' (' + tagCounts[tag] + ')'
            })
        }
    }

    // plot meta
    const chartLayout_ = deepCopy(chartLayout);
    chartLayout_.yaxis.title.text = 'Segment amplitude (dB)';

    return [dataAnnots, chartLayout_]
}



/* EVENTS */
// detections update
function runDetFunc(detDur='null') {
    const dtNow = getDtNowString();
    const elem = getElem('modelSelectDropdown');
    if (elem.options == undefined) { return; }
    const modelId = elem.value;
    if (modelId == '') { console.log('Select model to run detector.'); return; }
    console.log('Updating detections at ' + dtNow + ' with modelId ' + modelId);
    startProgressBar();
    return setDetections(formatFEDt(dtNow), modelId, detDur)
        .then(n => {
            stopProgressBar();
            updateDetTimeline();
        });
}


getElem('modelSelectDropdown').addEventListener('change', ev => runDetFunc());


// data timeline and macroseg events
function setTimelineEvents() {
    // timeline click event (macroseg gen and display)
    getElem('detectionsDiv').on('plotly_click', function(data) {
        const macrosegDt = data.points[0].x;
        updateMacrosegAfterTimelineClick(macrosegDt);
    });
}

function updateMacrosegAfterTimelineClick(macrosegDt) {
    getMacrosegData(macrosegDt)
        .then(obj => {
            const audFpath = "/" + obj.macrosegFpath;
            updateMacrosegAudio(audFpath);
            return fetchAudio(audFpath);
        })
        .then(macrosegWf => drawMacroseg(macrosegDt, macrosegWf));
}

function updateMacrosegAudio(audFpath) {
    // update audio element
    let text = '';
    text += `<audio controls="" src=${audFpath} type="audio/mpeg"`;
    text += ` style="width:1090px; padding-left: 32px;"`;
    text += `></audio>`
    audioElem.innerHTML = text;
}

function drawMacroseg(macrosegDt, macrosegWf) {
    // data
    const wfMax = maxAbsArray(macrosegWf);
    const dataForPlot = [{
        x: initArrayRange(macrosegWf.length),
        y: macrosegWf,
        type: 'line',
        name: 'audio'
    }];

    // layout
    const chartLayout_ = deepCopy(chartLayout);
    let time = splitDt(macrosegDt)[1];
    time = time.substring(0, time.length - 4); // remove milliseconds
    chartLayout_.title.text = 'Waveform at ' + time;
    chartLayout_.showlegend = false;
    chartLayout_.xaxis.range = [0, macrosegWf.length];
    chartLayout_.yaxis.range = [-1.05 * wfMax, 1.05 * wfMax];
    chartLayout_.xaxis.title.text = "Sample index";

    // draw
    setDisplay('audioMacrosegDiv', 'block');
    updatePlot('audioMacrosegDiv', dataForPlot, chartLayout_);
}


// data collection toggle
getElem('dataCollectCheckbox').addEventListener('change', processDataCollectSwitch);

function processDataCollectSwitch() {
    const switchState = getElem('dataCollectCheckbox').checked;
    setDataCaptureState(switchState);
}








