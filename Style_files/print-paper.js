function printAuthor(author) {
    if (author.name.search("Wenting") >= 0) document.write("<i><b>"+author.name+"</b></i>");
    else document.write("<i>"+author.name+"</i>");
};

function printPaper(paper){
    document.write("<div class=title display=block style=\"cursor:pointer\" onclick=\"return toggleAbstract(\'abs-" + paper.id + "\');\"><b>");
    document.write(paper.name);
    document.writeln("</b>");

    if (typeof paper.tags != "undefined"){
        var i;
        document.write(" [");
        for (i = 0; i < paper.tags.length; i++){
            document.write(paper.tags[i].name);
            if (i == paper.tags.length - 2)
                document.write("+");
            else if (i != paper.tags.length - 1) document.write("+");
        }
        document.writeln("]");
    }
	document.writeln("</div>");
    if (typeof paper.coauthors != "undefined"){
        var i;
        document.write("<div> ");
        for (i = 0; i < paper.coauthors.length; i++){
            printAuthor(paper.coauthors[i]);
            if (i == paper.coauthors.length - 2)
                document.write(" and ");
            else if (i != paper.coauthors.length - 1) document.write(", ");
        }
        document.writeln("</div>");
    }
    if (paper.conference == "Manuscript") document.writeln("<i>Manuscript</i>");
    else if (paper.conference == "Dissertation") document.writeln("<i>Ph.D. Dissertation</i>");
    else
        document.writeln("In <b><i>" + paper.conference + "</i></b>");
    if (typeof paper.note == "string") document.write("&emsp;<i><b>("+paper.note+")</b></i>");
    if (typeof paper.confVersion == "string") document.write("&emsp;<a href=\""+paper.confVersion+"\">[pdf]</a>"); 
    if (typeof paper.arxiv == "string") document.write("&emsp;<a href=\""+paper.arxiv+"\">[arXiv]</a>");
    if (typeof paper.journal == "string") document.write("<br>In <b><i>"+paper.journal+"</i></b>");
    //if (typeof paper.slides == "string") document.write("&emsp;<a href=\""+paper.slides+"\">[slides]</a>");
    if (typeof paper.talk == "string") document.write("&emsp;<a href=\""+paper.talk+"\">[talk]</a>");
    if (typeof paper.paperAbstract == "string") {
        document.writeln("<div class=\"abstract\" id=\'abs-" + paper.id + "\'>");
        document.writeln("<p><b>Abstract</b></p>");
        document.writeln(paper.paperAbstract);
        document.writeln("</div>");
    };
};
