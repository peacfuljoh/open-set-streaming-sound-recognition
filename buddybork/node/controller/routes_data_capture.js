

const miscUtils = require('../utils/misc_utils.js');
const constants = require('../constants_app.js');
const errorUtils = require('../utils/error_utils.js');


const express = require('express');

const router = express.Router();


function loadDataCapInfo() {
    return miscUtils.readJSON(constants.DATA_CAP_STATE_JSON_PATH);
}

function saveDataCapInfo(dataCapInfo) {
    miscUtils.writeJSON(constants.DATA_CAP_STATE_JSON_PATH, dataCapInfo);
}

router.post('/set/:state', (req, res) => {
    const state = req.params.state;

    errorUtils.validateDataCollectState(res, state);

    dataCapInfo = loadDataCapInfo();
    const capState_0 = dataCapInfo.capture_state;

    if      (state == 'true')  { dataCapInfo.capture_state = true; }
    else if (state == 'false') { dataCapInfo.capture_state = false; }
    else                       { return; }
    const capState_1 = dataCapInfo.capture_state;

    console.log('Setting data capture state: ' + capState_0 + ' -> ' + capState_1);
    saveDataCapInfo(dataCapInfo);
});

router.post('/get', (req, res) => {
    const state = req.params.state;

    dataCapInfo = miscUtils.readJSON(constants.DATA_CAP_STATE_JSON_PATH);
    res.json(dataCapInfo);
});

router.all('/*', (req, res) => errorUtils.checkErr(req, res, false, 'router_data_capture'))


module.exports = router;