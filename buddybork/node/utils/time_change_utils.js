/*
Note: duplicated functionality in static/misc_utils.js
Note: the EST-EDT crossover points aren't exactly right here because the setHours() method
*/

// EST or EDT
function getTimeChangeStatus(dt) {
    // beginning of EDT
    const dateEDTStart = new Date(dt);
    dateEDTStart.setMonth(2, 12); // month, day (March 12)
    dateEDTStart.setHours(3, 0, 0, 0); // hour, minute, second, ms (7 AM) --> this set op depends on the current time zone!!!

    // end of EDT
    const dateEDTEnd = new Date(dt);
    dateEDTEnd.setMonth(10, 5); // month, day (Nov 5)
    dateEDTEnd.setHours(2, 0, 0, 0); // hour, minute, second, ms (6 AM) --> this set op depends on the current time zone!!!

    // determine if EST or EDT
    return (dt >= dateEDTStart && dt <= dateEDTEnd) ? 'EDT' : 'EST';
}

// negated offset between UTC and EST/EDT in minutes
function getTimeZoneOffset(dt) {
    return dt.getTimezoneOffset(); // uses local time as reference
//    return getTimeChangeStatus(dt) == 'EST' ? 5 : 4;
}


module.exports = {
    getTimeZoneOffset
}
