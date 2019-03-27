
class Node:
    def __init__(self, val=None):
        self.left = None
        self.right = None 
        self.val = val

    def height(self):
        lh, rh = 0, 0
        if self.left != None:
            lh = self.left.height()
        if self.right != None:
            rh = self.right.height()

        return max(lh, rh) + 1

    def get_mask(self, word):
        return self.get_mask_r(word, [])

    def get_mask_r(self, word, _mask):
        if type(self.val) == str:
            if self.val == word:
                return _mask 
            else:
                return None 
        else:
            mask = self.left.get_mask_r(word, _mask+[0])
            if mask != None:
                return mask
            mask = self.right.get_mask_r(word, _mask+[1])
            return mask


    def __str__(self):
        return str(self.left)+'  '+str(self.val)+' '+str(self.right)

    def get_word_r(self, mask, h):
        if type(self.val) == str:
            return self.val
        if mask[h]>=0.5:
            return self.right.get_word_r(mask, h+1)
        else:
            return self.left.get_word_r(mask, h+1)
    
    def get_word(self, mask):
        return self.get_word_r(mask, 0)

    def prob_word_r(self, mask, word, prob, h):
        # print(prob)
        if type(self.val) == str:
            # print(self.val, word)
            if self.val == word:
                return True, prob
            else:
                return False, None
        rprob = mask[h]
        lprob = 1 - rprob

        found, new_prob = self.left.prob_word_r(mask, word, prob*lprob, h+1)
        if found:
            return True, new_prob
        found, new_prob = self.right.prob_word_r(mask, word, prob*rprob, h+1)
        if found:
            return True, new_prob 
        
        return False, None
    
    def prob_word(self, Y, word):
        found, prob = self.prob_word_r(Y, word, 1, 0)
        # print(found, prob)
        return prob