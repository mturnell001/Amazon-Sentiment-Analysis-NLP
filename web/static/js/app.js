//app.js

const analyzeButton = d3.select('#analyze-button');
analyzeButton.on("click", function(d) {
    d3.select('#output-list').html('');
    d3.select('#output-list').html('<img src="http://bestanimations.com/Science/Gears/Gears-04-june.gif"></img>');
    getSentiment();
});

function getSentiment() {
    const review = getReview();
    d3.json(`/api/${review}`).then(data => showSentiment(data));
}

function getReview() {
    const review = d3.select('#review-text').property('value');
    return review;
}

function showSentiment(json_obj) {
    var ul = d3.select('#output-list');
    
    ul.html('');

    var selection = ul.selectAll('li') //not working properly
        .data(Object.entries(json_obj));
    
    selection.enter()
        .append('li')
        .text(d => `${d[0]}: ${d[1]}`)
        .merge(selection);
        
    selection.exit().remove();
}   