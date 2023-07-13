/*
    Back-end error handling utils.
*/

const moment = require('moment');

const miscUtils = require('./misc_utils.js');


const TRAINER_OPS_BASE = ['predict', 'fit']
const TRAINER_OPS_RM = ['delete'];
const TRAINER_OPS = TRAINER_OPS_BASE.concat(TRAINER_OPS_RM)
const TRAINER_DATA_KEYS = ['days', 'tags', 'model_id', 'model_name'];
const TRAINER_DATA_KEYS_RM = ['model_id'];
const DATA_COLLECT_STATES = ['true', 'false'];


function checkErr(req, res, valid=false, func=null) {
    if (!valid) {
        console.log('An error was detected for ' + func);
        res.render('error');
    }
}


function validateDayString(res, day) {
    const valid = moment(day, "YYYY-MM-DD", true).isValid(); // 'true' is for exact format matching
    checkErr(null, res, valid, 'validateDayString');
}

function validateDtString(res, dt) {
    const valid = [
        'YYYY-MM-DD_HH:mm:ss.SSS',
        'YYYY-MM-DD HH:mm:ss.SSS',
        'YYYY-MM-DD_HH:mm:ss.SS',
        'YYYY-MM-DD HH:mm:ss.SS'].some(fmt => moment(dt, fmt, true).isValid())
//    moment(dt, 'YYYY-MM-DD_HH:mm:ss.SSS', true).isValid() ||
//                    moment(dt, 'YYYY-MM-DD HH:mm:ss.SSS', true).isValid();
    checkErr(null, res, valid, 'validateDtString');
}

function validateModelId(res, modelId) {
    const valid = (modelId.length == 13) && miscUtils.isNumeric(modelId);
    checkErr(null, res, valid, 'validateModelId');
}

function validateDur(res, dur) {
    const valid = (dur == 'null') || miscUtils.isNumeric(dur);
    checkErr(null, res, valid, 'validateDur');
}

function validateTrainerOp(res, op) {
    const valid = TRAINER_OPS.includes(op);
    checkErr(null, res, valid, 'validateTrainerOp');
}

function validateTrainerData(res, op, data) {
    const dataKeys = TRAINER_OPS_RM.includes(op) ? TRAINER_DATA_KEYS_RM : TRAINER_DATA_KEYS;
    const valid = miscUtils.arrEquality(Object.keys(JSON.parse(data)), dataKeys);
    checkErr(null, res, valid, 'validateTrainerData');
}

function validateDataCollectState(res, state) {
    const valid = DATA_COLLECT_STATES.includes(state);
    checkErr(null, res, valid, 'validateDataCollectState');
}




module.exports = {
    validateDayString,
    validateDtString,
    validateModelId,
    validateDur,
    validateTrainerOp,
    validateTrainerData,
    validateDataCollectState,
    checkErr
}
