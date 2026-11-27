% Encoding: UTF-8
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}\relax
\ProvidesClass{include/seminar}

\DeclareOption{a4paper}{\PassOptionsToClass{\CurrentOption}{scrartcl}}
\DeclareOption{12pt}{\PassOptionsToClass{\CurrentOption}{scrartcl}}
\DeclareOption{oneside}{\PassOptionsToClass{\CurrentOption}{scrartcl}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrartcl}}
\ExecuteOptions{a4paper,12pt,oneside}
\ProcessOptions\relax
\LoadClass{scrartcl}

\RequirePackage[headsepline]{scrlayer-scrpage}

\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[ngerman]{babel}
\RequirePackage[autostyle=true,german=quotes]{csquotes}
\RequirePackage{bibgerm}
\RequirePackage{lmodern}
\RequirePackage{graphicx}
\RequirePackage{textcomp}

\RequirePackage[protrusion=true, expansion=true, final]{microtype}
\usepackage[backend=bibtex,bibencoding=utf-8,style=alphabetic-verb]{biblatex}

\makeatletter

%% Header of each page
\lohead{\sem@author}
\rohead{\sem@shorttitle}
\pagestyle{scrheadings}

%% Title page
\newcommand{\Author}[1]{\newcommand{\sem@author}{#1}}
\newcommand{\Studiengang}[1]{\newcommand{\sStudiengang}{#1}}

\newcommand{\Title}{\@dblarg\seminar@title}
\long\def\seminar@title[#1]#2{
    \def\sem@title{#2}
    \def\sem@shorttitle{#1}
}

\newcommand{\Prof}[1]{\newcommand{\prof}{#1}}
\newcommand{\Chair}[1]{\newcommand{\chair}{#1}}

\newcommand{\makeAuthor}{\author{\sem@author\\\sStudiengang}}
\newcommand{\makeTitle}{\title{\sem@title}}

\makeatother

% Head of title page
\titlehead{
    \parbox{7cm}{
        Betreuer:\\
		\prof \vspace{1ex}\\
        \chair \vspace{1ex}\\Universität Augsburg
    }
    \hfill
    \parbox{2.2cm}{
        \includegraphics[width=2.0cm]{include/img/unilogo}
    }
}

\endinput
