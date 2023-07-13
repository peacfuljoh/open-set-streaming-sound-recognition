/*
    Back-end utils for running model-related ops.
*/

const pyUtils = require('../utils/py_utils.js');
const miscUtils = require('../utils/misc_utils.js');
const constants = require('../constants_app.js');


function runModelOp(mode, params) {
    return pyUtils.runPyOp(constants.PYMODEL_PATH, [mode, params]);
}

function getModelInfo() {
    return miscUtils.readJSON(constants.MODEL_JSON_FPATH);
}


module.exports = {
    runModelOp,
    getModelInfo
}