
const express = require('express');

const router = express.Router();

const modelUtils = require('../utils/model_utils.js');
const renderer = require('./renderers.js');
const errorUtils = require('../utils/error_utils.js');


function mainTrainerRoute(req, res) {
    renderer.render(res, 'trainer');
}

router.get('/', mainTrainerRoute);
router.post('/', mainTrainerRoute);


router.post('/model/:op/:data', (req, res) => {
    const op = req.params.op;
    const data = req.params.data;
    errorUtils.validateTrainerOp(res, op);
    errorUtils.validateTrainerData(res, op, data);
    modelUtils.runModelOp(op, data)
        .then(obj => {
//            console.log(op)
//            console.log(data)
//            console.log(obj)
            return res.json(obj);
        })
})

function getModelInfo(res) {
    res.json(modelUtils.getModelInfo());
}

router.post('/modelInfo', (req, res) => getModelInfo(res));
router.get('/modelInfo', (req, res) => getModelInfo(res));

router.all('/*', (req, res) => errorUtils.checkErr(req, res, false, 'router_trainer'))

module.exports = router;
