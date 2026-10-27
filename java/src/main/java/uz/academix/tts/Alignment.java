package uz.academix.tts;

import java.util.*;

final class Alignment {
    private final int first,last,eos,size,hardCap;
    private final List<float[]> history=new ArrayList<>();
    private int step,textPosition,completedAt=-1;
    private final List<Integer> previous=new ArrayList<>();
    Alignment(int first,int last,int eos){this.first=first;this.last=last;this.eos=eos;size=last-first;hardCap=Math.max(150,size*25);}
    void apply(float[] scores,float[] attention,int queries,int keys,int token) {
        int[] heads={2,15,11};int begin=step==0?last:0;
        float[] current=null;
        for(int q=begin;q<queries;q++) {
            current=new float[size];
            for(int t=0;t<size;t++) {
                if(t>step)continue;
                for(int l=0;l<3;l++)current[t]+=attention[((l*16+heads[l])*queries+q)*keys+first+t]/3f;
            }
            history.add(current);
        }
        if(current==null)throw new IllegalStateException("Missing alignment frame");
        int pos=0;for(int t=1;t<size;t++)if(current[t]>current[pos])pos=t;
        if(pos-textPosition>-4 && pos-textPosition<7)textPosition=pos;
        if(completedAt<0 && textPosition>=size-3)completedAt=history.size();
        boolean longTail=false,repetition=false;
        if(completedAt>=0) {
            for(int t=Math.max(0,size-3);t<size;t++) {
                double sum=0;for(int r=completedAt;r<history.size();r++)sum+=history.get(r)[t];
                if(sum>=5)longTail=true;
            }
            if(size>5) {
                double sum=0;
                for(int r=completedAt;r<history.size();r++) {
                    float max=0;for(int t=0;t<size-5;t++)max=Math.max(max,history.get(r)[t]);sum+=max;
                }
                repetition=sum>5;
            }
        }
        previous.add(token);int n=previous.size();
        boolean repeated=n>=3&&previous.get(n-1).equals(previous.get(n-2));
        boolean capped=step>=hardCap;
        if(!capped && pos<size-3 && size>5)scores[eos]=-32768;
        if(longTail||repetition||repeated||capped){Arrays.fill(scores,-32768);scores[eos]=32768;}
        step++;
    }
}
