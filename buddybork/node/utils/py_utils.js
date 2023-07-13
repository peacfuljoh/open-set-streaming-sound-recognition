/*
    Back-end Python sub-process utils.
*/

const spawn = require('child_process').spawn;

const constants = require('../constants_app.js');


function runPyOp(scriptName, args) {
    // - scriptName is a string
    // - args is a JSONified dictionary
    // - python process must return string beginning with 'pyout:' in order for promise to resolve, remaining contents
    //     of string are passed back to FE
    // - returns a dictionary/object
    return new Promise((success, nosuccess) => {
        const pyprog = spawn(constants.PYTHON_ENV_PATH, [scriptName, constants.PYTHON_SRC_PATH].concat(args));
        pyprog.on('error', (err) => console.error('Failed to start subprocess.'));
        pyprog.stdout.on('data', data => {
//            console.log('<----\n' + data.toString() + '\n---->')
            for (s of data.toString().split('\n')) {
                if      (s.substring(0, 6) == 'pyout:') { success(JSON.parse(s.substring(6))); }
                else if (s != '')                       { console.log(s); }
            }
        });
        pyprog.stderr.on('data', data => nosuccess(data));
    }).catch(err => console.error(err.toString()));
}

module.exports = {
    runPyOp
}