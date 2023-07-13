
const express = require('express');

const miscUtils = require('../utils/misc_utils.js');


const router = express.Router();


function serveMeta(res) {
    const cjs = miscUtils.readJSON();
    for (key in cjs) {
        if (key.includes('COMMENT') || (key == 'DB_CONFIG')) {
            delete cjs[key];
        }
    }
    return res.json(cjs);
}


router.post('/', (req, res) => serveMeta(res));
router.get('/', (req, res) => serveMeta(res));

router.all('/*', (req, res) => errorUtils.checkErr(req, res, false, 'router_meta'))

module.exports = router;