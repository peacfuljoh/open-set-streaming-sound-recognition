/*
    Back-end general-purpose constants.
*/

const fs = require('fs');
const path = require('node:path');

// load paths and constants
const PATHS_FPATH = process.env.BB_PATHS;
const paths = JSON.parse(fs.readFileSync(PATHS_FPATH));
const CONST_JSON_PATH = path.join(process.env.BB_ROOT, 'constants.json');
const constantsGlobal = JSON.parse(fs.readFileSync(CONST_JSON_PATH));

// load local config
function getLocalConfig(configPath, defaultObj) {
    if (fs.existsSync(configPath)) {
        return JSON.parse(fs.readFileSync(configPath));
    }
    console.log('Could not find config file: ' + configPath);
    return defaultObj;
}
const DB_CONFIG = getLocalConfig(paths.DB_CONFIG_PATH, {});
const NET_CONFIG = getLocalConfig(paths.NET_CONFIG_PATH, {HUB_IP: '0', SERVER_PORT: 0});


// export all constants
module.exports = {
    IP_ADDRESS: NET_CONFIG.HUB_IP,
    SERVER_PORT: NET_CONFIG.HUB_NODE_PORT,
    DB_CONFIG: DB_CONFIG,
    ANNOT_SEG_HALFWIDTH: 2,
    CONST_JSON_PATH: CONST_JSON_PATH, //'/home/nuc/web-apps/buddybork/constants.json',
    LEN_HASH: 30,
    HASH_CHARS: `abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()`,
    PYDETECTION_PATH: 'py/pydetection.py',
    PYMODEL_PATH: 'py/pymodel.py',
    DETECTION_DUR: 6 * 60, // minutes
    DET_BATCH_SIZE: 1000,
    MODEL_JSON_FPATH: path.join(paths.MODEL_DIR, 'models.json'), //'/home/nuc/buddybork_model/models.json'
    PYTHON_ENV_PATH: paths.PYTHON_ENV_PATH,
    PYTHON_SRC_PATH: process.env.BB_ROOT, //"/home/nuc/web-apps/buddybork"
    MAX_QUERY_LOG_LEN: 500,
    DATA_CAP_STATE_JSON_PATH: paths.CAPTURE_STATE_JSON_PATH,
    ASSETS_DIR: paths.ASSETS_DIR
}

//let tzoffset = (new Date()).getTimezoneOffset() * 60000; // offset in milliseconds
