In an effort to save others from the pain and suffering that I have gone through to get the LaTeX and the BYU Thesis/Dissertations package to do what I wanted, I cleaned it up and made some changes. It tries to be a small, neat, clean tutorial (which I don't know how good of a job it does), but it provides all of the necessary details and a good structure to start from. The byustyle pacakge follows the model of the hyperref package (it has a few optional parameters and then a setup function to configure everything). I've tried to put comments around all of the commands to make it easier to make it do what you want but the general run down is this.

Hopefully, this will help stop people from pulling out as many hairs as I have.
Dave Johansen

*****Added March 2008 by Brian Winder:
Information About LaTeX:
Go to http://en.wikipedia.org/wiki/LaTeX for a good overview of the LaTeX document preparation system.


*******Added June 2012 by Brian Jensen
For a good overview of using LaTeX, see the document "The Not So Short Introduction to LaTeX2e," which is included in the distribution of the LaTeX thesis template by the Mechanical Engineering Department. The filename is LaTeXIntroduction.pdf.

Installing a LaTeX Editor:
1) For Windows, TeXnicCenter is a good open source editor for LaTeX.  It requires MikTeX, so in order to use it, you will first need to download and install MikTeX (http://www.miktex.org/ or look for it on SourceForge.net) and then install TeXnicCenter (http://www.texniccenter.org/).  Do not try to install the full distribution of MikTeX when you install the program, as that will take up extra space on your hard drive for many packages that will never get used.  It is better to install the bare minimum and let MikTeX install packages as it needs them (TeXnicCenter will tell MikTeX to install any missing packages when you compile a document that requires them).  
2) For Mac, I have heard that TeXShop (http://www.uoregon.edu/~koch/texshop/) is a great editor and I have seen people use it with great results, though I do not have any personal experience with it.

*******

Template Use Instructions (updated March 2008):
1) Open the master.tex file in your editor of choice and compile it the way it is (set the output profile in TeXnicCenter to "LaTeX => PDF" so that you get a PDF).  In some editors, you may have to compile it more than once for all the figure, table and bibliography references to be formed properly (TeXnicCenter will give you lots of warnings until it works out all of the references).
2) The output is a thesis/dissertation that is in the format that BYU wants the final version to be.  In many editors, compiling the document and viewing it are two separate operations, so if you compile it and the PDF file does not show up, find the button for viewing it.  You can also open the master.pdf file that is created in the same directory as the master.tex file.  If you do that, make sure you close it before trying to compile again, since the computer needs to write over it and it will not compile if the file is already open.
3) Dave Johansen compiled this template on the EE Unix/Linux boxes, the CS Linux boxes, and his own machine (MikTeX with Windows XP) and he included the .sty sheets needed for it to compile in all those settings. Ideally, you would get rid of them and have all those packages in your install of LaTeX and you'd keep them up to date, but they're there so that it can compile right out of the door for everyone. The files that would probably be installed on most LaTeX installations but are included here for completeness are asmems4.bst, chngpage.sty, keyval.tex, xkeyval.sty, and xkeyval.tex.

Other Little Tips (updated March 2008):
1) You'll need to set all of the name properties or you can make them use defaults (big ugly text that says it's missing) by using the option usedefaultnames.
2) The copyrightyear will just set itself to the current year, but you can change that if you want.
3) You can turn off any part of the BYU header using the commands in the \byustylesetup{} on line 28 through 38.
4) Let's say that you just want to print one chapter for some hard copy editing but you don't want to screw up all of the chapter numbers and bibTeX stuff, then you'll notice some commands starting on line 121. Those commands will make LaTeX only put that chapter in the generated document but leave all of the numbering and such untouched. You can also do multiple chapters by doing stuff like \includeonly{synchronization,results}. It's a nice little trick that I've really liked using.
5) If you're using TeXnicCenter, make sure you create a project with master.tex as the main file.  This allows easy navigation through the entire document and through your references (helps avoid the warnings when you misspell a citation).

Printing Tips:
1) When you print your PDF make sure that "Page Scaling" is set to "None". The default value is "Fit to Printer Margins" and this will cause everything to be not quite the right size.
2) Also, check to make sure that your document is the right size (8.5" x 11"). It has been discovered that several LaTeX installations default to generating a document that is on A4 paper and this will cause your margins to be off throughout your document when you print. To fix this follow these steps (with MikTeX):
	2a) Go to the "dvipdfm\config" folder in your MikTeX installation ("C:\texmf" by default) and edit the "config" file in a text editor
	2b) Change "p a4" to "p letter"

	2c) Go to the "dvips\config" folder in your MikTeX installation ("C:\texmf" by default) and edit the "config.ps" file in a text editor
	2d) Make sure that "letterSize" is the first entry in the list of paper sizes by changing the order from

        @ A4size 594.99bp 841.99bp
       	@+ ! %%DocumentPaperSizes: a4
        @+ %%PaperSize: A4

        @ letterSize 8.5in 11in
        @+ ! %%DocumentPaperSizes: Letter

	to

	@ letterSize 8.5in 11in
        @+ ! %%DocumentPaperSizes: Letter

        @ A4size 594.99bp 841.99bp
        @+ ! %%DocumentPaperSizes: a4
        @+ %%PaperSize: A4
	


This file and the template package have been modified since Dave's original writing.
Last modified: March 2008