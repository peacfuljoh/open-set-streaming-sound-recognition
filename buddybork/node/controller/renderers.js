/*
    Renders pug templates with inserted data based on route specified.
*/


content = require('./renderer_content.js');


const helpText = {
    'index': content.helpTextIndex,
    'annotator': content.helpTextAnnotator,
    'trainer': content.helpTextTrainer,
    'login': content.helpTextLogin,
}


function render(res, route) {
    const data = {};
    if (route in helpText) { data['helpContent'] = helpText[route]; }
    res.render(route, data);
}


module.exports = {
    render
}

