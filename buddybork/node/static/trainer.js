/*
    Front-end code for handling interaction on Model Builder page.
*/


/* CONSTANTS */
const groupModeLabels = ["None", "Train (R)", "Test (E)", "Background Train (Br)", "Background Test (Be)"];
const groupModeValues = ['N', 'R', 'E', 'Br', 'Be'];

const BAR_HEIGHT_NO_ANNOTS = 10;


/* STATE */
const groupModeByDay = {x: [], y: [], text: []};
const annotCountsByDay = {};

let numTagsSelect;
let lastDeleteClick = new Date();


/* INIT */
updateTimeline();
updateTagSelect();
setupMenus();



/* MISC */
function resetGroupModeByDay() {
    groupModeByDay.x = [];
    groupModeByDay.y = [];
    groupModeByDay.text = [];
}

function getCheckboxElems() {
    return initArrayRange(numTagsSelect).map(i => getElem('tagCheckbox' + i));
}

function resetTagsSelect() {
    getCheckboxElems().forEach(elem => elem.checked = false);
}

function updateTagSelectWithModelTags(tags) {
    getCheckboxElems().forEach(elem_i => { elem_i.checked = tags != null && tags.includes(elem_i.name); })
}

function resetTimelineAndTagsSelect() {
    resetGroupModeByDay();
    updateTimeline();
    resetTagsSelect();
}



/* PLOTS */
async function updateTimeline() {
    // get all annot info
    const annots = await getAnnotDataAllGroupedByDay(); // dict of array of dt objects
    const tagsInfo = await getTags(); // dict w/ list of strings (label-value dicts) and colors (list of arrays)
    const dtLims = await getDataLims();
    let allDays = await getAllDataDays();

    if (maxDayDisplay != null) {
        cropByDayRange(annots, 'obj', null, maxDayDisplay)
        cropByDayRange(allDays, 'arrOfDays', null, maxDayDisplay)
        cropByDayRange(dtLims, 'dtLims', null, maxDayDisplay)
    }

    for (day of allDays) {
        if (Object.keys(annots).includes(day)) { annotCountsByDay[day] = annots[day].length; }
        else                                   { annotCountsByDay[day] = BAR_HEIGHT_NO_ANNOTS; }
    }

    drawTimeline(annots, tagsInfo, dtLims, allDays);
    setTimelineEvents();
}

function drawTimeline(annots, tagsInfo, dtLims, allDays) {
    // get tag info
    tagsInfo.colors = tagsInfo.colors.map(arr => { return arr.map(v => Math.min(255, v / 0.85)); });
    let [tags, tagColors] = getTagsAndColorsFromTagsInfo(tagsInfo);
    const numTags = tags.length;

//    console.log(annots)
//    console.log(dtLims)
//    console.log(allDays)

    perm = makePermutationDet(numTags);
    tags = perm.map(i => tags[i]);
    tagColors = perm.map(i => tagColors[i]);

    // accumulate segmented bar chart data
    const days = Object.keys(annots);
    const dataForPlot = [];
    tags.forEach((tag, j) => {
        const counts_j = [];
        const days_j = [];
        for (const day of days) {
            const count = annots[day].reduce((acc, annot) => acc + Number(annot['tag'] == tag), 0);
            if (count) {
                counts_j.push(count);
                days_j.push(day);
            }
        }
        dataForPlot.push({
            x: days_j,
            y: counts_j,
            type: 'bar',
            marker: { color: tagColors[j] },
            name: tag
        });
    });
    const daysNoAnnots = [];
    allDays.forEach(day => {
        if (!days.includes(day)) {
            daysNoAnnots.push(day)
        }
    })
    dataForPlot.push({
        x: daysNoAnnots,
        y: initArray(daysNoAnnots.length, BAR_HEIGHT_NO_ANNOTS),
        type: 'bar',
        marker: { color: 'w' },
        name: 'no annots'
    });
//    console.log(groupModeByDay)
    dataForPlot.push({
        x: groupModeByDay.x,
        y: groupModeByDay.y,
        type: 'line',
        mode: 'text',
        name: 'groupMode',
        text: groupModeByDay.text,
        textposition: 'top',
        showlegend: false
    });

    // plot meta
    const chartLayout_ = deepCopy(chartLayout);
    chartLayout_.title.text = "Annotations";
    chartLayout_.xaxis.title.text = "Day";
    chartLayout_.xaxis.range = padDtLims(dtLims);
    chartLayout_.yaxis.title.text = "Annotation count";
    chartLayout_.barmode = 'stack';
//    chartLayout_.showlegend = false;

    updatePlot('daysAnnotsDiv', dataForPlot, chartLayout_);
}



/* TAG SELECT */
function updateTagSelect() {
    getTags()
        .then(tagsInfo => {
            const [tags, tagColors] = getTagsAndColorsFromTagsInfo(tagsInfo);

            numTagsSelect = tags.length; // global

            let text = '';
            text += `Training Tags Select<pre></pre>`;
            for (let i = 0; i < tags.length; i++) {
                text += `<label for="tagCheckbox${i}" style="font-size: 12px;">`;
                text += `<input type="checkbox" id="tagCheckbox${i}" name="${tags[i]}" value="yes"`;
                text += ` style="width: 12px; height: 12px;"`;
                text += `>${tags[i]}`;
                text += `</label>`;
                text += `<br>`;
            }

            getElem("tagsSelectDiv").innerHTML = text;
        });
}



/* MENUS */
function setupMenus() {
    setupGroupModeMenu();
    setupModelSelectMenu();
}

function setupGroupModeMenu() {
    // group mode select
    const dropdownOptions = initArrayRange(groupModeLabels.length).map(i => {
        return {label: groupModeLabels[i], value: groupModeValues[i]};
    });
    setDropdownOptions("groupModeDropdown", dropdownOptions);
}

function setupModelSelectMenu(val=null) {
    // model select
    getModelInfo()
        .then(modelInfo => {
            const modelIds = Object.keys(modelInfo);
            let dropdownOptions = [{label: '', value: ''}];
            const modelIdOptions = modelIds.map(id_ => {
                return {label: modelInfo[id_].model_name, value: id_};
            });
            dropdownOptions = dropdownOptions.concat(modelIdOptions);
            setDropdownOptions("modelSelectDropdown", dropdownOptions);
            if (val != null && modelIds.includes(val)) {
                getElem('modelSelectDropdown').value = val;
            }
        });
}




/* EVENTS */
// timeline events
function setTimelineEvents() {
    getElem('daysAnnotsDiv').on('plotly_click', function(data) {
        const data0 = data.points[0];
        const day = data0.x;
//        const tag = data0.data.name;

        const mode = getElem("groupModeDropdown").value;
        removeElemFromGroupModes(day);
        if (mode != 'N') {
            addElemToGroupModes(day, mode);
        }
        updateTimeline();
    });
}

function addElemToGroupModes(day, mode) {
    groupModeByDay.x.push(day);
    groupModeByDay.y.push(annotCountsByDay[day] + 1);
    groupModeByDay.text.push(mode);
}

function removeElemFromGroupModes(day) {
    const idx = groupModeByDay.x.indexOf(day);
    if (idx != -1) {
        groupModeByDay.x.splice(idx, 1);
        groupModeByDay.y.splice(idx, 1);
        groupModeByDay.text.splice(idx, 1);
        updateTimeline();
    }
}

// model fit/pred events
getElem('fitModelButton').addEventListener('click', ev => modelFunc('fit'));
getElem('predModelButton').addEventListener('click', ev => modelFunc('pred'));

function getModelInfoSelections(mode) {
    const numDays = groupModeByDay.x.length;
    if (numDays == 0) { return null; }

    const days = {};
    initArrayRange(numDays).forEach(i => {
        const day = groupModeByDay.x[i];
        const group = groupModeByDay.text[i];
        days[day] = group;
    })
    tags = getCheckedTags();
    modelId = getElem("modelSelectDropdown").value;
    const modelName = (mode == 'fit') ? prompt("Model name:", "") : '';

    return {
        days: days,
        tags: tags,
        model_id: modelId,
        model_name: modelName
    };
}

function getCheckedTags() {
    const tags = [];
    for (let i = 0; i < numTagsSelect; i++) {
        const elem_i = getElem('tagCheckbox' + i);
        if (elem_i.checked) {
            tags.push(elem_i.name);
        }
    }
    return tags;
}

function modelFunc(mode) {
    modelInfo = getModelInfoSelections(mode);
    if (modelInfo == null) { return }
    if (mode == 'fit')  {
        func = fitModel;
        if (modelInfo.model_name == '') { return }
    }
    if (mode == 'pred') {
        func = predModel;
        if (!Object.values(modelInfo.days).some(m => m == 'E' || m == 'Be') || modelInfo.model_id == '') { return }
    }
    startProgressBar();
    func(JSON.stringify(modelInfo))
        .then(result => {
//            console.log(result)
            stopProgressBar();
            if (mode == 'fit') {
                setupModelSelectMenu(result.modelId);
            }
            if (mode == 'pred') {
                updatePerfDiv(result);
            }
        });
}

function updatePerfDiv(result) {
    // update model performance results plot
    const confMat = result.confMat
    const tagsTrain = result.tagsTrain;
    const tagsTest = result.tagsTest;

    const dataToPlot = [{
        z: confMat,
        colorscale: 'YlGnBu',
        type: 'heatmap',
        showscale: false,
        name: ''
    }];
    for (let i = 0; i < tagsTrain.length; i++) {
        const newData = {
            x: initArrayRange(tagsTest.length),
            y: initArray(tagsTest.length, i),
            type: 'line',
            mode: 'text',
            text: confMat[i].map(n => n.toString()),
            textposition: 'middle',
            textfont: { color: 'red', size: 15 },
            showlegend: false,
            name: ''
        };
        dataToPlot.push(newData);
    }

    const chartLayout_ = deepCopy(chartLayout);
    chartLayout_.title.text = "Confusion matrix";
    chartLayout_.xaxis.title.text = "Test tag";
    chartLayout_.yaxis.title.text = "Train tag";
    chartLayout_.yaxis.autorange = "reversed";

    chartLayout_.yaxis.tickvals = initArrayRange(tagsTrain.length);
    chartLayout_.yaxis.ticktext = tagsTrain;
    chartLayout_.yaxis.tickfont = {size: 10};

    chartLayout_.xaxis.tickvals = initArrayRange(tagsTest.length);
    chartLayout_.xaxis.ticktext = tagsTest;
    chartLayout_.xaxis.tickfont = {size: 10};

    const opts_ = {
        modeBarButtonsToAdd: ["select2d"],
        modeBarButtonsToRemove: ["toImage"],
        displaylogo: false
    };

    getElem('perfDiv').style.width = 200 + 50 * tagsTest.length;

    updatePlot('perfDiv', dataToPlot, chartLayout_, opts_);
}

// menu select events
getElem("modelSelectDropdown").addEventListener('change', modelSelectFunc);

function modelSelectFunc() {
    const modelId = getElem("modelSelectDropdown").value;
    resetGroupModeByDay();
    if (modelId == '') {
        resetTimelineAndTagsSelect();
        return;
    }
    getModelInfo()
        .then(modelInfo => {
            const model_ = modelInfo[modelId];
            model_.days_bg.forEach(day => addElemToGroupModes(day, 'Br'));
            model_.days_train.forEach(day => addElemToGroupModes(day, 'R'));
            updateTagSelectWithModelTags(model_.tags);
            updateTimeline();
        })
}

// delete model event
getElem("deleteModelButton").addEventListener('click', deleteModelFunc);

function deleteModelFunc() {
    const newClick = new Date();
    if (newClick - lastDeleteClick > 200) { lastDeleteClick = newClick; return }
    const modelId = getElem("modelSelectDropdown").value;
    if (modelId == '') { return }
    deleteModel(JSON.stringify({model_id: modelId}))
        .then(result => {
            setupModelSelectMenu();
            resetTimelineAndTagsSelect();
        })
}
