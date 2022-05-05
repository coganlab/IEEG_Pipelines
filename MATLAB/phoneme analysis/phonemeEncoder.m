function [binClass,phonCVClass,phonIndClass] = phonemeEncoder(phonLabel)
    switch(phonLabel)
        case 'a'
            binClass = 1;
            phonCVClass = 1;
            phonIndClass = 1;
        case 'ae'
            binClass = 1;
            phonCVClass = 1;
            phonIndClass = 2;
        case 'i'
            binClass = 1;
            phonCVClass = 2;
            phonIndClass = 3;
        case 'u'
            binClass = 1;
            phonCVClass = 2;
            phonIndClass = 4;
        case 'b'
            binClass = 2;
            phonCVClass = 3;
            phonIndClass = 5;
        case 'p'
            binClass = 2;
            phonCVClass = 3;
            phonIndClass = 6;
        case 'v'
            binClass = 2;
            phonCVClass = 3;
            phonIndClass = 7;
        case 'g'
            binClass = 2;
            phonCVClass = 4;
            phonIndClass = 8;
        case 'k'
            binClass = 2;
            phonCVClass = 4;
            phonIndClass = 9;
    end
end