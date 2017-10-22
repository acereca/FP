// Usage
// .L batchproc.C
// single(scintillatorNo)

#include <TROOT.h>
#include <TH1D.h>
#include <TFile.h>
#include <TLegend.h>
#include <TCanvas.h>

using namespace std;

int nLayers = 6;

void single(int layerToUse, const char *filename) {
    TCanvas *c1 = new TCanvas;

    TFile *f = new TFile(filename, "READ");
    TH1D *a = (TH1D*) f->Get(Form("a%d", layerToUse));
    
    a -> Draw("e");
    TH1D *x = (TH1D*) f->Get(Form("x%d", layerToUse));

    TH1D *h = (TH1D*) f->Get("h8");
    Double_t Nstart = 0;
    for (int i = layerToUse+1; i <= 5; i++) {
        Nstart += h->GetBinContent(i + 1);
    }

    cout << "Nstart = " << Nstart << endl;

    Double_t Nstop = h -> GetBinContent(layerToUse+1);
    cout << "Nstop = " << Nstop << endl;

    Double_t scaleFactor = Nstop/Nstart;
    cout << "s = " << scaleFactor << endl;
    x -> Scale(scaleFactor);
    x -> SetLineColor(kRed);
    x -> Draw("esame");
    TH1D *s = new TH1D(*x);
    a -> Add(s, -1.);

    TLegend* leg = new TLegend(.78,.6,.98,.7);
    leg -> AddEntry(a, "Zerfaelle", "l");
    leg -> AddEntry(x, "Nachpulse (skaliert)", "l");
    leg -> Draw();

    a -> GetYaxis() -> SetRangeUser(-4000,4000);
}


void batchProcessUpwards(const char *filename){
    for (int i = 1; i < nLayers-1; i++){
        single(i, filename);
    }
}