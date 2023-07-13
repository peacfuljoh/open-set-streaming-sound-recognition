/*
    Front-end code for handling interaction on Annotator page.
*/

let annotModeOpts = ["none", "insert", "remove"]
annotModeOpts = annotModeOpts.map(s => strToOption(s));

let macrosegDt; // datetime at start of currently active macroseg (used for inferring datetime on click)
let macrosegFirstSamp = null; // first samp in annot interval
let macrosegWf; // macroseg audio waveform data


const audioElem = getElem('audioPlayback');
const annotModeDropdownElem = getElem('annotModeDropdown');
const tagDropdownElem = getElem('tagDropdown');
const macrosegLeftNavElem = getElem("macrosegLeftNav");
const macrosegRightNavElem = getElem("macrosegRightNav");
const macrosegLeftNavFastElem = getElem("macrosegLeftNavFast");
const macrosegRightNavFastElem = getElem("macrosegRightNavFast");



/* init */
updateDaysTimeline();
updateAnnotTimeline();



/* Functions */

// plot update
function updateTimelines(day=null) {
    updateAnnotTimeline();
    updateSegTimeline(day);
}


// update timeline of day-by-day segment counts
async function updateDaysTimeline() {
    const counts = await getRawDataDayCounts();
    let dtLims = await getDataLims();

    if (maxDayDisplay != null) {
        cropByDayRange(counts, 'obj', null, maxDayDisplay)
        cropByDayRange(dtLims, 'dtLims', null, maxDayDisplay)
    }

    drawDaysTimeline(counts, dtLims);
    setDaysTimelineEvents();
}

function drawDaysTimeline(counts, dtLims) {
    // plot data
    const days = Object.keys(counts);
    const counts_ = days.map(d => counts[d]); // counts as an array
    const dataForPlot = [{
        x: days,
        y: counts_,
        type: 'scatter',
        marker: { color: 'blue', size: 10 },
        mode: 'markers',
        name: ''
    }];
    days.forEach((day, i) => {
        dataForPlot.push({
            x: [day, day],
            y: [0, counts_[i]],
            type: 'line',
            line: { color: 'blue' },
            mode: 'lines',
            name: ''
        });
    });

    // plot meta
    const chartLayout_ = deepCopy(chartLayout);
    chartLayout_.title.text = "Database summary";
    chartLayout_.xaxis.title.text = "Day";
    chartLayout_.xaxis.range = padDtLims(dtLims);
    chartLayout_.yaxis.range = [0, 1.1 * Math.max(...dataForPlot[0].y)];
    chartLayout_.yaxis.title.text = "Daily segment count";
    chartLayout_.showlegend = false;

    updatePlot('daysTimelineDiv', dataForPlot, chartLayout_);
}


// update annot timeline
async function updateAnnotTimeline() {
    const tagsInfo = await getTags(); // keys: tags, colors
    const annots = await getAnnotDataAll();
    const dtLims = await getDataLims();

    if (maxDayDisplay != null) {
        cropByDayRange(annots, 'arrOfAnnots', null, maxDayDisplay)
        cropByDayRange(dtLims, 'dtLims', null, maxDayDisplay)
    }

    drawAnnotTimeline(annots, tagsInfo, dtLims);
    setAnnotTimelineEvents();
}

function drawAnnotTimeline(annots, tagsInfo, dtLims) {
    //
    [tags, tagColors] = getTagsAndColorsFromTagsInfo(tagsInfo);

    // plot data
    const dataAnnots = [];
    const tagCounts = {};
    for (tag of tags) {
        const tagIdx = tags.indexOf(tag);
        const annotIdxs = [];
        annots.forEach((annot, idx) => { (annot.tag == tag) ? annotIdxs.push(idx) : null; });
        tagCounts[tag] = annotIdxs.length;
        dataAnnots.push({
            x: annotIdxs.map(idx => annots[idx].datetime_start),
            y: initArray(tagCounts[tag], tagIdx),
            type: 'scatter',
            marker: { color: tagColors[tagIdx], size: 5 },
            mode: 'markers',
            name: ''
        })
    }

    // plot meta
    const chartLayout_ = deepCopy(chartLayout);
    chartLayout_.title.text = "Annotations";
    chartLayout_.xaxis.title.text = "Day";
    chartLayout_.xaxis.range = padDtLims(dtLims);
    chartLayout_.yaxis.tickvals = initArrayRange(tags.length);
    chartLayout_.yaxis.ticktext = tags.map(tag => tag + ' (' + tagCounts[tag] + ')');
    chartLayout_.yaxis.tickfont = {size: 10};
    chartLayout_.yaxis.range = [-0.5, tags.length];
//    chartLayout_.yaxis.title.text = "Tag";
    chartLayout_.showlegend = false;

    updatePlot('annotTimelineDiv', dataAnnots, chartLayout_);
}


// update timeline
function updateSegTimeline(day=null) {
    const day_ = (day == null) ? splitDt(macrosegDt)[0] : day;
    console.log(day_)
    getRawDataForDay(day_)
        .then(data => updateSegTimelinePlot(data, day_))
}

async function updateSegTimelinePlot(data, day) {
    // data: object with datetime and max_amp arrays
    const chartLayout_ = deepCopy(chartLayout);
    chartLayout_.title.text = "Data segments for " + day;
    chartLayout_.yaxis.title.text = 'Segment amplitude (dB)';
    chartLayout_.showlegend = false;

    let dataForPlot = [];

    const numData = data.datetime.length;
    if (numData) {
        // tag info
        const tagsInfo = await getTags(); // keys: tags, colors
        [tags, tagColors] = getTagsAndColorsFromTagsInfo(tagsInfo);

        // first seg dt in this macroseg
        const dt0 = data.datetime[0];

        // plot meta
        chartLayout_.xaxis.range = getDateRange(dt0);

        // get annots for this day
        const annots = await getAnnotDataForDay(day);
        const minAmp = Math.min(...data.max_amp);
        const maxAmp = Math.max(...data.max_amp);
        const annotDataForPlot = annots.map(annot => {
            return {
                x: initArray(2, annot.datetime_start),
                y: [ampToDb(minAmp), ampToDb(maxAmp)],
                type: 'line',
                mode: 'lines',
                line: { color: tagColors[tags.indexOf(annot.tag)], width: 1 },
                zorder: 0,
                name: ''
            }
        });

        // gather data for plot
        dataForPlot = [{
            x: data.datetime,
            y: data.max_amp.map(val => ampToDb(val)),
            type: 'scatter',
            mode: 'markers',
            zorder: 10,
            name: ''
        }].concat(annotDataForPlot);
    }

    updatePlot('timelineDiv', dataForPlot, chartLayout_);

    setTimelineEvents();
}

// update macroseg plot
function updateMacrosegPlot() {
    getAnnotDataForMacrosegPlot(macrosegDt)
        .then(annots => drawMacrosegAndAnnots(annots))
        .then(n => setMacrosegEvents())
}

async function drawMacrosegAndAnnots(annots) {
    // data
    const wfMax = maxAbsArray(macrosegWf);
    const annotData = await makeAnnotsData(annots, wfMax);
    const dataForPlot = [{
        x: initArrayRange(macrosegWf.length),
        y: macrosegWf,
        type: 'line',
        name: 'audio'
    }].concat(annotData);

    // layout
    let chartLayout_ = deepCopy(chartLayout);
    let time = splitDt(macrosegDt)[1];
    time = time.substring(0, time.length - 4); // remove milliseconds
    chartLayout_.title.text = 'Waveform at ' + time;
    chartLayout_.showlegend = false;
    chartLayout_.xaxis.range = [0, macrosegWf.length];
    chartLayout_.yaxis.range = [-1.05 * wfMax, 1.05 * wfMax];
    chartLayout_.xaxis.title.text = "Sample index";

    const marginLeft = 62;
    const marginRight = 0;
    chartLayout_ = {
        ...chartLayout_,
        autosize: false,
        width: window.innerWidth * 0.82 - (marginLeft + marginRight), // same percentage as in pug template
        height: 340,
        margin: {
            l: marginLeft,
            r: marginRight,
            b: 45,
            t: 50,
            pad: 0
        }
    }

    // draw
    setDisplay('audioMacrosegDiv', 'block');
    updatePlot('audioMacrosegDiv', dataForPlot, chartLayout_, {displayModeBar: false});

    return 0
}

function clearMacroseg() {
    const ids = [
        'audioPlayback', 'audioMacrosegDiv', 'tagDropdown',
        'tagDropdownLabel', 'annotModeDropdown', 'annotModeDropdownLabel',
        'macrosegLeftNav', 'macrosegRightNav', 'macrosegLeftNavFast',
        'macrosegRightNavFast', 'macrosegNavLabel'
    ];
    for (id_ of ids) {
        setDisplay(id_, 'none');
    }
}

async function makeAnnotsData(annots, wfMax) {
    const tagsInfo = await getTags(); // keys: tags, colors
    [tags, tagColors] = getTagsAndColorsFromTagsInfo(tagsInfo);

    const annotsData = annots.map((annot, i) => {
        const color_i = tagColors[tags.indexOf(annot.tag)];
        return {
            x: [annot.samp_start, annot.samp_end],
            y: initArray(2, 0.8 * wfMax * (1 - i / 10)),
            type: 'line',
            mode: 'lines+markers+text',
            name: 'annot' + i,
            marker: { color: color_i },
            line: { color: color_i },
            text: initArray(2, annot.tag),
            textposition: 'top'
        }
    })
    return annotsData
}

// annotator dropdown menus, buttons, etc.
function updateAnnotatorMenu() {
    // tags menu
    getTags()
        .then(tagsInfo => {
            setDropdownOptions("tagDropdown", tagsInfo.tags);
            setDisplay("tagDropdown", 'inline');
            setDisplay("tagDropdownLabel", 'inline');
        })

    // mode menu
    setDropdownOptions("annotModeDropdown", annotModeOpts);
    setDisplay("annotModeDropdown", 'inline');
    setDisplay("annotModeDropdownLabel", 'inline');

    // left and right navigation buttons
    setDisplay('macrosegNavLabel', 'inline');
    setDisplay('macrosegLeftNav', 'inline');
    setDisplay('macrosegRightNav', 'inline');
    setDisplay('macrosegLeftNavFast', 'inline');
    setDisplay('macrosegRightNavFast', 'inline');
}



/* Set up event handlers */
// daily count timeline events
function setDaysTimelineEvents() {
    getElem('daysTimelineDiv').on('plotly_click', function(data) {
        const day = data.points[0].x;
        processSelectDate(day)
    })
}

function processSelectDate(day) {
    updateSegTimeline(day);
    clearMacroseg();
}


// annot timeline events
function setAnnotTimelineEvents() {
    getElem('annotTimelineDiv').on('plotly_click', function(data) {
        const data0 = data.points[0];
        const dt = data0.x;
        const day = splitDt(dt)[0];

        updateSegTimeline(day);
        macrosegDt = dt;
        updateMacrosegAfterTimelineClick();
    })
}

// data timeline and macroseg events
function setTimelineEvents() {
    // timeline click event (macroseg gen and display)
    getElem('timelineDiv').on('plotly_click', function(data) {
        macrosegDt = data.points[0].x;
        console.log(macrosegDt)
        updateMacrosegAfterTimelineClick();
    });
}

function updateMacrosegAfterTimelineClick() {
    getMacrosegData(macrosegDt) // get macroseg from server for the current dt
        .then(obj => {
            const audFpath = "/" + obj.macrosegFpath;
            updateMacrosegAudio(audFpath); // update audio element
            return fetchAudio(audFpath); // fetch audio signal
        })
        .then(wf => {
            macrosegWf = wf; // save audio signal in client-side global
            if (getDisplay("tagDropdown") == "none") {
                updateAnnotatorMenu(); // show macroseg controls
            }
            updateMacrosegPlot(); // update plot with waveform and annotations
        })
}

function updateMacrosegAudio(audFpath) {
    // update audio element
    setDisplay('audioPlayback', 'block');
    let text = '';
    text += `<audio controls="" src=${audFpath} type="audio/mpeg"`;
    text += ` style="width:115%; padding-left:1%;"`;
    text += `></audio>`
    audioElem.innerHTML = text;
}

function setMacrosegEvents() {
    // annotation click event
    macrosegFirstSamp = null;

    getElem('audioMacrosegDiv').on('plotly_click', function(data) {
        const data0 = data.points[0];
        const name = data0.data.name;
        const mode = annotModeDropdownElem.value;

        let processFunc;
        if (name.includes('audio') && mode == 'insert') {
            processFunc = processAnnotInsert;
        }
        else if (name.includes('annot') && mode == 'remove') {
            processFunc = processAnnotRemove;
        }
        else {
            return
        }
        processFunc(data0)
            .then(bool => {
                if(bool) { updateTimelines(); }
            })
    });
}


/* Annotation event handling */
function processAnnotInsert(data0) {
    // collect info for annot
    const samp = data0.x;

    if (!macrosegFirstSamp) {
        macrosegFirstSamp = samp;
        return new Promise((res, rej) => { return res(false); });
    }

    // issue annot action
    const tag = tagDropdownElem.value;
    const dt = formatFEDt(macrosegDt);

    const r = '/annotator/annotInsert/' + tag + '/' + dt + '/' + macrosegFirstSamp + '/' + samp;
    macrosegFirstSamp = null;

    return annotRouteHandler(r)
}

function processAnnotRemove(data0) {
    // issue annot action
    const lineData = data0.data;
    const tag = lineData.text[0];
    const dt = formatFEDt(macrosegDt);
    const samp = lineData.x[0]; // left-most sample

    const r = '/annotator/annotRemove/' + tag + '/' + dt + '/' + samp;
    macrosegFirstSamp = null;

    return annotRouteHandler(r)
}

function annotRouteHandler(r) {
    return fetchHandler(r)
        .then(obj => {
            console.log(obj); // show command that was run on server
            updateMacrosegPlot();
        })
        .then(n => true)
}


// set left and right navigation events
macrosegLeftNavElem.addEventListener('click', ev => updateMacrosegDtHandler(-1))
macrosegRightNavElem.addEventListener('click', ev => updateMacrosegDtHandler(1))
macrosegLeftNavFastElem.addEventListener('click', ev => updateMacrosegDtHandler(-5))
macrosegRightNavFastElem.addEventListener('click', ev => updateMacrosegDtHandler(5))

async function updateMacrosegDtHandler(multSeg) {
    await updateMacrosegDt(multSeg);
    updateMacrosegAfterTimelineClick();
}

function updateMacrosegDt(multSeg) {
    return getMeta()
        .then(meta => {
            const segDurMs = meta.BUFF_DUR * 1000;
            const dt = (new Date(macrosegDt));
            dt.setMilliseconds(dt.getMilliseconds() + multSeg * segDurMs);
            macrosegDt = convertDateToString(dt);
        })
}

