
const express = require('express');

const detectUtils = require('../utils/detector_utils.js');
const miscUtils = require('../utils/misc_utils.js');
const errorUtils = require('../utils/error_utils.js');

const router = express.Router();


// POST method to get latest detections
router.post('/get/:dt', (req, res) => {
    let dt = req.params.dt;
    errorUtils.validateDtString(res, dt);
    dt = miscUtils.formatBEDt(dt);
    detectUtils.getDetections(dt)
        .then(obj => res.json(obj))
})

// POST method to trigger detection using specified model
router.post('/set/:dt/:modelId/:dur', (req, res) => {
    let dt = req.params.dt;
    const modelId = req.params.modelId;
    let dur = req.params.dur;

    errorUtils.validateDtString(res, dt);
    errorUtils.validateModelId(res, modelId);
    errorUtils.validateDur(res, dur);

    dt = miscUtils.formatBEDt(dt);
    dur = (dur == 'null') ? null : Number(dur);

    detectUtils.setDetections(dt, modelId, dur)
        .then(obj => res.json(obj))
})

router.all('/*', (req, res) => errorUtils.checkErr(req, res, false, 'router_detector'))

module.exports = router;
