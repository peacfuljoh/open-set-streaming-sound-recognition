

/* Functions */
const db = require('../db/database_utils.js');
const miscUtils = require('../utils/misc_utils.js');
const errorUtils = require('../utils/error_utils.js');


function getDtLims() {
    return db.selectQuery('MAX(datetime) as datetime_max, MIN(datetime) as datetime_min', 'raw')
        .then(result => {
            const tmin = result[0].datetime_min;
            const tmax = result[0].datetime_max;
            return [
                miscUtils.convertDateToString(tmin),
                miscUtils.convertDateToString(tmax)
            ]
        })
}

function getRawDataDayCounts() {
    return db.selectQuery('datetime', 'raw')
        .then(result => {
            const dts = result.map(row => row.datetime);
            return miscUtils.getDailyDtCounts(dts)
        })
}

function getTimelineData(day) {
    return db.selectQuery("*", 'raw', `WHERE date(datetime) = '${day}'`) // datetimes return in GMT timezone
        .then((result) => {
            return {
                "datetime": result.map(row => miscUtils.convertDateToString(row.datetime)),
                "max_amp": result.map(row => row.max_amp)
            };
        })
        .catch((err) => console.log(err));
}

function getAllDays() {
    return db.selectQuery("DISTINCT ON (datetime::date) *", "raw", "ORDER BY datetime::date")
        .then(result => result.map(row => miscUtils.splitDt(miscUtils.convertDateToString(row.datetime))[0]))
}



/* Routes */
const express = require('express');

const router = express.Router();


// get data datetime limits
router.post('/dtLims', (req, res) => {
    getDtLims()
        .then(obj => res.json(obj))
})

function getDayCounts(res) {
    getRawDataDayCounts()
        .then(obj => res.json(obj))
}

// get timeline data counts
router.post('/data/dayCounts', (req, res) => getDayCounts(res));
//router.get('/data/dayCounts', (req, res) => getDayCounts(res));

// get timeline data for a specified day
router.post('/data/timeline/:day', (req, res) => {
    const day = req.params.day;
    errorUtils.validateDayString(res, day);
    getTimelineData(day)
        .then(obj => res.json(obj))
})

// get all days in raw seg database
router.post('/allDays', (req, res) => {
    getAllDays()
        .then(obj => res.json(obj))
})

router.all('/*', (req, res) => errorUtils.checkErr(req, res, false, 'router_raw'))

module.exports = router;
