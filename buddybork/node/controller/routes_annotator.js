
const express = require('express');

const db = require('../db/database_utils.js');
const annotUtils = require('../utils/annotator_utils.js');
const miscUtils = require('../utils/misc_utils.js');
const renderer = require('./renderers.js');
const errorUtils = require('../utils/error_utils.js');


function mainAnnotatorRoute(req, res) {
    renderer.render(res, 'annotator');
}


const router = express.Router();

router.get('/', mainAnnotatorRoute);
router.post('/', mainAnnotatorRoute);


// generate macroseg and get path to it
router.post('/data/macroseg/:dt', (req, res) => {
    const dt = req.params.dt;
    errorUtils.validateDtString(res, dt);
    annotUtils.makeMacroSeg(dt)
        .then(obj => res.json(obj))
})

// get all annotations
router.post('/data/annots', (req, res) => {
    annotUtils.getAnnotData()
        .then(obj => res.json(obj))
})

// get all annotations, grouped by day
router.post('/data/annots/groupByDay', (req, res) => {
    annotUtils.getAnnotData()
        .then(annots => annotUtils.groupAnnotsByDay(annots))
        .then(obj => res.json(obj))
})

// get annotations within a macroseg
router.post('/data/annots/macroseg/:dt', (req, res) => {
    const dt = req.params.dt;
    errorUtils.validateDtString(res, dt);
    annotUtils.getAnnotData(dt)
        .then(obj => res.json(obj))
})

// get annotations on a day
router.post('/data/annots/day/:day', (req, res) => {
    const day = req.params.day;
    errorUtils.validateDayString(res, day);
    annotUtils.getAnnotData(null, day)
        .then(obj => res.json(obj))
})

// get annotation tags
router.post('/tags', (req, res) => {
    annotUtils.getAnnotatorTags()
        .then(obj => res.json(obj))
})

// process annotation event (insert)
router.post('/annotInsert/:tag/:dt/:samp0/:samp1', (req, res) => {
    const tag = req.params.tag;
    let dt = req.params.dt;
    const samp0 = req.params.samp0;
    const samp1 = req.params.samp1;

    errorUtils.validateDtString(res, dt);

    dt = miscUtils.formatBEDt(dt);

    annotUtils.processAnnotEvent(tag, 'insert', dt, samp0, samp1)
        .then(obj => res.json(obj))
})

// process annotation event (remove)
router.post('/annotRemove/:tag/:dt/:samp', (req, res) => {
    const tag = req.params.tag;
    let dt = req.params.dt;
    const samp = req.params.samp;

    errorUtils.validateDtString(res, dt);

    dt = miscUtils.formatBEDt(dt);

    annotUtils.processAnnotEvent(tag, 'remove', dt, samp)
        .then(obj => res.json(obj))
})

router.all('/*', (req, res) => errorUtils.checkErr(req, res, false, 'router_annotator'))

module.exports = router;