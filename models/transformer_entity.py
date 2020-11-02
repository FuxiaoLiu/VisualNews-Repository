        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        guide = self.p.expand(batch_size, self.hid_dim).unsqueeze(1).expand(batch_size, Q.size(1), self.hid_dim)
        guide = self.tanh(self.fc_g(guide))
        Q = torch.mul(Q, guide)

        #print('Q:', )


        #print('V:', V.size())
        #print('past:', guide.size())

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        #x = torch.matmul(self.dropout(attention), V)
        x = torch.matmul(attention, V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        #x1 = self.tanh(self.fc_g1(x))
         #= torch.mul(Q, x1)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)


        # x = [batch size, query len, hid dim]

        x = torch.cat([x, Q], dim=2)

        x1 = self.sigmoid(self.l1(x))
        x2 = self.l2(x)

        x = torch.mul(x1, x2)



        return x, attention
